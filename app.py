"""
KneeScan — knee MRI analysis workstation (Gradio).

Upgrades over the first version:
  * Interactive slice viewer (scroll, window/level, Grad-CAM opacity, key-slice jump,
    clickable filmstrip of the 16 slices the model actually saw)
  * Live decision-threshold slider (re-scores instantly, no re-inference)
  * Case library with filters (click a row to analyse it), random-case button
  * Session history table + CSV export, standalone HTML report export
  * Model card tab (architecture, results, limitations, checkpoint status)
  * Honest labelling: each score says whether it comes from a dedicated model,
    multi-plane fusion, or a proxy estimate
  * Cached models / labels / transforms / Grad-CAM (no per-click reloading)

Requires Gradio >= 4.20 (tested against the 4.x / 5.x API).
"""
import os
import re
import io
import time
import base64
import random
import tempfile
import datetime as dt
from html import escape

import cv2
import numpy as np
import pandas as pd
import torch
import gradio as gr
from PIL import Image

from dataset import MRNetDataset
from models import MRNetLSTM, GradCAM

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Override without editing code:  set MRNET_ROOT=D:\data\MRNet-v1.0
DATASET_ROOT = os.environ.get(
    'MRNET_ROOT',
    r'C:\Users\USER\OneDrive\Desktop\DEEP LEARNING\project\MRNet-v1.0',
)

NUM_SLICES = 16
PLANES = ('axial', 'sagittal')
PLANE_LABEL = {'axial': 'Axial PD', 'sagittal': 'Sagittal T2'}
TASKS = ('acl', 'meniscus', 'abnormal')
BORDERLINE_BAND = 0.07          # |p - threshold| below this is shown as "borderline"

# Checkpoints. Meniscus / abnormal heads are optional: if the file exists the card
# uses a real model, otherwise it falls back to a clearly-labelled proxy estimate.
CKPT = {
    ('acl', 'axial'): 'best_lstm.pth',
    ('acl', 'sagittal'): 'best_lstm_sagittal.pth',
    ('meniscus', 'axial'): 'best_lstm_meniscus.pth',
    ('abnormal', 'axial'): 'best_lstm_abnormal.pth',
}

# Logistic fusion of per-plane ACL probabilities (from your fitted fusion model).
FUSION = dict(w_sag=1.052, w_ax=3.394, bias=-2.1)

# Shown on the Model Card tab. Update these if you retrain.
MODEL_CARD = dict(
    backbone='ResNet18 slice encoder (early layers frozen), 224x224 input',
    head='Bidirectional LSTM + slice attention, 16 slices per exam',
    loss='Weighted BCE (pos_weight ~ 4.43)',
    data='Stanford MRNet (1,130 train / 120 validation exams)',
    metric_task='ACL tear, axial plane, official validation set (n = 120)',
    auc_baseline=0.7823,
    auc_model=0.7904,
)

HIST_COLS = ['Time', 'Case', 'ACL %', 'Meniscus %', 'Abnormal %', 'ACL ground truth', 'Latency (ms)']

# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
_MODELS = {}
_CAMS = {}
_TRANSFORMS = {}
_LABELS = None


def get_model(task, plane):
    """Return a loaded model for (task, plane) or None if no checkpoint exists."""
    key = (task, plane)
    if key in _MODELS:
        return _MODELS[key]
    path = CKPT.get(key)
    model = None
    if path and os.path.exists(path):
        model = MRNetLSTM(hidden_dim=128, num_layers=1, freeze_early=True).to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
    _MODELS[key] = model
    return model


def get_transform(plane):
    if plane not in _TRANSFORMS:
        try:
            ds = MRNetDataset(DATASET_ROOT, split='valid', plane=plane, task='acl', num_slices=NUM_SLICES)
        except Exception as exc:  # dataset folder missing / unreadable
            raise gr.Error(f'Could not initialise the preprocessing pipeline: {exc}')
        _TRANSFORMS[plane] = ds.transform
    return _TRANSFORMS[plane]


def get_labels():
    """{task: {'0001': 0/1, ...}} for the validation split, read once."""
    global _LABELS
    if _LABELS is None:
        _LABELS = {}
        for task in TASKS:
            path = os.path.join(DATASET_ROOT, f'valid-{task}.csv')
            table = {}
            if os.path.exists(path):
                df = pd.read_csv(path, header=None, names=['id', 'label'])
                table = {str(int(i)).zfill(4): int(v) for i, v in zip(df['id'], df['label'])}
            _LABELS[task] = table
    return _LABELS


def list_case_ids():
    ids = sorted(get_labels()['acl'].keys())
    return ids or ['1130', '1131', '1132', '1133', '1134']


def checkpoint_status():
    return {f'{t} / {p}': os.path.exists(path) for (t, p), path in CKPT.items()}


# ----------------------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------------------
def load_volume(path):
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise gr.Error(f'Could not read the .npy file: {exc}')
    if arr.ndim != 3:
        raise gr.Error(f'Expected a 3-D volume shaped (slices, H, W); got shape {arr.shape}.')
    if arr.shape[0] < 1:
        raise gr.Error('The volume contains no slices.')
    return arr


def sample_indices(n):
    return np.linspace(0, n - 1, NUM_SLICES, dtype=int) if n > NUM_SLICES else np.arange(n)


def to_tensor(vol, plane):
    idx = sample_indices(vol.shape[0])
    tf = get_transform(plane)
    slices = [tf(np.stack([s] * 3, axis=-1)) for s in vol[idx]]
    return torch.stack(slices, dim=0).unsqueeze(0).to(DEVICE), idx


def predict(model, tensor):
    with torch.no_grad():
        return float(torch.sigmoid(model(tensor)).item())


def sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def run_gradcam(plane, model, tensor, idx):
    key = ('acl', plane)
    if key not in _CAMS:
        _CAMS[key] = GradCAM(model)
    with torch.enable_grad():
        heat, pos = _CAMS[key].generate(tensor)
    if hasattr(heat, 'detach'):
        heat = heat.detach().cpu().numpy()
    heat = np.asarray(heat, dtype=np.float32)
    if heat.ndim > 2:
        heat = heat.squeeze()
    span = float(heat.max() - heat.min())
    heat = (heat - heat.min()) / span if span > 1e-8 else np.zeros_like(heat)
    pos = int(min(max(int(pos), 0), len(idx) - 1))
    return dict(heat=heat, sampled_pos=pos, raw_idx=int(idx[pos]))


def compute_scores(probs, tensors):
    """Turn per-plane probabilities into the three report targets."""
    p_ax, p_sag = probs.get('axial'), probs.get('sagittal')
    scores = {}

    # --- ACL: the one target this checkpoint set is actually trained for ---
    if p_ax is not None and p_sag is not None:
        logit = FUSION['w_sag'] * p_sag + FUSION['w_ax'] * p_ax + FUSION['bias']
        scores['acl'] = dict(prob=sigmoid(logit), source='Multi-plane fusion')
    else:
        plane = 'axial' if p_ax is not None else 'sagittal'
        scores['acl'] = dict(prob=probs[plane], source=f'Single plane ({plane}) - no fusion')

    # --- Meniscus / abnormal: dedicated head if a checkpoint exists, else proxy ---
    dedicated = {}
    if 'axial' in tensors:
        for task in ('meniscus', 'abnormal'):
            m = get_model(task, 'axial')
            if m is not None:
                dedicated[task] = predict(m, tensors['axial'])

    if 'meniscus' in dedicated:
        scores['meniscus'] = dict(prob=dedicated['meniscus'], source='Dedicated model')
    else:
        a = p_ax if p_ax is not None else p_sag
        s = p_sag if p_sag is not None else p_ax
        scores['meniscus'] = dict(prob=float(a * 0.92 + 0.05 * s), source='Proxy estimate (experimental)')

    if 'abnormal' in dedicated:
        scores['abnormal'] = dict(prob=dedicated['abnormal'], source='Dedicated model')
    else:
        worst = max(scores['acl']['prob'], scores['meniscus']['prob'])
        scores['abnormal'] = dict(
            prob=float(min(0.999, max(0.05, worst * 0.95 + 0.08))),
            source='Proxy estimate (experimental)',
        )
    return scores


# ----------------------------------------------------------------------------
# Rendering: slice viewer
# ----------------------------------------------------------------------------
def render_core(ctx, plane, idx, level, window, show_cam, opacity, size=512, annotate=True):
    vol = ctx['volumes'][plane]
    n = vol.shape[0]
    idx = int(min(max(idx, 0), n - 1))

    lo, hi = ctx['range'][plane]
    x = np.clip((vol[idx].astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    w_lo, w_hi = level - window / 2.0, level + window / 2.0
    x = np.clip((x - w_lo) / max(w_hi - w_lo, 1e-6), 0, 1)
    gray = (x * 255).astype(np.uint8)
    rgb = np.stack([gray] * 3, axis=-1)

    cam = ctx['cams'].get(plane)
    on_key = bool(cam) and idx == cam['raw_idx']
    if show_cam and on_key:
        heat = cv2.resize(cam['heat'], (gray.shape[1], gray.shape[0]))
        colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_JET),
                               cv2.COLOR_BGR2RGB)
        rgb = cv2.addWeighted(rgb, 1.0 - opacity, colored, opacity, 0)

    h, w = rgb.shape[:2]
    scale = size / float(max(h, w))
    rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)

    if annotate:
        font = cv2.FONT_HERSHEY_SIMPLEX

        def put(text, org, color=(235, 240, 245)):
            cv2.putText(rgb, text, org, font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(rgb, text, org, font, 0.5, color, 1, cv2.LINE_AA)

        H, W = rgb.shape[:2]
        put(f'{PLANE_LABEL[plane]}   {idx + 1}/{n}', (10, 20))
        put(f'W {window:.2f}  L {level:.2f}', (10, H - 12), (160, 175, 190))
        tags = []
        if idx in set(ctx['indices'].get(plane, [])):
            tags.append('MODEL INPUT')
        if on_key:
            tags.append('KEY SLICE')
        if tags:
            text = '  '.join(tags)
            tw = cv2.getTextSize(text, font, 0.5, 1)[0][0]
            put(text, (W - tw - 10, 20), (90, 200, 255) if on_key else (170, 185, 200))
    return rgb


def build_filmstrip(ctx, plane, level, window):
    if not ctx or plane not in ctx['indices']:
        return []
    cam = ctx['cams'].get(plane)
    strip = []
    for pos, raw_idx in enumerate(ctx['indices'][plane]):
        img = render_core(ctx, plane, raw_idx, level, window, False, 0.0, size=160, annotate=False)
        star = ' (key)' if cam and cam['sampled_pos'] == pos else ''
        strip.append((img, f'#{int(raw_idx) + 1}{star}'))
    return strip


def viewer_bundle(ctx, plane, level, window, show_cam, opacity, focus_idx=None):
    """slider update, viewer image, filmstrip for a plane."""
    n = ctx['volumes'][plane].shape[0]
    cam = ctx['cams'].get(plane)
    idx = focus_idx if focus_idx is not None else (cam['raw_idx'] if cam else n // 2)
    img = render_core(ctx, plane, idx, level, window, show_cam, opacity)
    return (
        gr.update(minimum=1, maximum=n, value=int(idx) + 1),
        img,
        build_filmstrip(ctx, plane, level, window),
    )


# ----------------------------------------------------------------------------
# Rendering: report
# ----------------------------------------------------------------------------
def verdict(p, thr):
    if abs(p - thr) <= BORDERLINE_BAND:
        return 'border', 'Borderline'
    return ('pos', 'Positive screen') if p > thr else ('neg', 'Negative screen')


VERDICT_COLOR = {'pos': '#ef4444', 'border': '#f59e0b', 'neg': '#10b981'}


def card_html(title, subtitle, score, gt, thr, custom, plane_probs=None):
    p = score['prob']
    state, label = verdict(p, thr)
    color = VERDICT_COLOR[state]

    if custom or gt is None:
        gt_line = '<span class="muted">Not available for this scan</span>'
    else:
        predicted_pos = p >= thr
        agree = predicted_pos == bool(gt)
        gt_line = (
            f'{"Positive" if gt == 1 else "Negative"} '
            f'<span class="chip {"chip-ok" if agree else "chip-bad"}">'
            f'{"agrees with model" if agree else "differs from model"}</span>'
        )

    proxy = score['source'].startswith('Proxy')
    src_chip = f'<span class="chip {"chip-proxy" if proxy else "chip-src"}">{escape(score["source"])}</span>'

    plane_chips = ''
    if plane_probs:
        plane_chips = '<div class="chips">' + ''.join(
            f'<span class="chip">{PLANE_LABEL[k]} {v * 100:.1f}%</span>' for k, v in plane_probs.items()
        ) + '</div>'

    return f"""
    <div class="target-card">
        <div class="target-card-header">
            <span class="target-title">{escape(title)}</span>
            <span class="status-badge badge-{state}"><span class="badge-dot"></span>{label}</span>
        </div>
        <div class="risk-display">
            <span class="risk-value" style="color:{color};">{p * 100:.1f}%</span>
            <span class="risk-label">model probability, threshold {thr * 100:.0f}%</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width:{p * 100:.1f}%; background-color:{color};"></div>
            <div class="thr-tick" style="left:{thr * 100:.1f}%;"></div>
        </div>
        {plane_chips}
        <div class="target-card-footer">
            <div class="meta-line"><span class="meta-key">Target</span> {escape(subtitle)}</div>
            <div class="meta-line"><span class="meta-key">Ground truth</span> {gt_line}</div>
            <div class="meta-line"><span class="meta-key">Source</span> {src_chip}</div>
        </div>
    </div>
    """


def render_report(ctx, thr):
    if not ctx:
        return EMPTY_REPORT
    custom = ctx['custom']
    gt = ctx['gt']
    sc = ctx['scores']
    plane_probs = {k: ctx['probs'][k] for k in PLANES if k in ctx['probs']}

    cards = (
        card_html('ACL ligament', 'Anterior cruciate ligament', sc['acl'], gt.get('acl'), thr, custom,
                  plane_probs if len(plane_probs) > 1 else None)
        + card_html('Meniscus', 'Medial and lateral menisci', sc['meniscus'], gt.get('meniscus'), thr, custom)
        + card_html('Overall abnormality', 'Joint space and effusion', sc['abnormal'], gt.get('abnormal'), thr, custom)
    )

    series = ', '.join(
        f'{PLANE_LABEL[p]} ({ctx["volumes"][p].shape[0]} slices)' for p in PLANES if p in ctx['volumes']
    )

    notes = []
    primary = ctx['primary']
    cam = ctx['cams'].get(primary)
    if cam:
        n = ctx['volumes'][primary].shape[0]
        notes.append(
            f'<li><b>Attention:</b> strongest Grad-CAM response on slice {cam["raw_idx"] + 1} of {n} '
            f'({PLANE_LABEL[primary]}). Open it in the viewer below.</li>'
        )
    if len(plane_probs) > 1:
        notes.append('<li><b>ACL strategy:</b> axial and sagittal probabilities are combined with a fitted logistic fusion.</li>')
    else:
        notes.append('<li><b>ACL strategy:</b> only one plane produced a prediction, so no multi-plane fusion was applied.</li>')
    if ctx['unused_planes']:
        names = ', '.join(PLANE_LABEL[p] for p in ctx['unused_planes'])
        notes.append(f'<li><b>Not scored:</b> no checkpoint found for {names}; the series is viewable but excluded from scoring.</li>')
    if any(s['source'].startswith('Proxy') for s in sc.values()):
        notes.append('<li><b>Proxy targets:</b> meniscus and overall abnormality are derived from the ACL-trained '
                     'model and are not independently validated. Treat them as exploratory.</li>')
    notes.append('<li><b>Use:</b> research prototype. Not a medical device and not a substitute for review by a radiologist.</li>')

    title = 'Uploaded scan' if custom else f'Case {escape(ctx["case"])}'
    return f"""
    <div class="report-container">
        <div class="report-header">
            <div>
                <span class="exam-title">{title}</span>
                <span class="exam-meta">{series} &nbsp;|&nbsp; analysed {escape(ctx['ts'])} in {ctx['latency_ms']:.0f} ms on {DEVICE.type.upper()}</span>
            </div>
        </div>
        <div class="targets-grid">{cards}</div>
        <div class="findings-box">
            <div class="findings-title">Summary</div>
            <ul class="findings-list">{''.join(notes)}</ul>
        </div>
    </div>
    """


EMPTY_REPORT = """
<div class="report-container empty-state">
    <div class="empty-title">No scan analysed yet</div>
    <div>Pick a case on the left (or upload a .npy volume) and choose <b>Analyse scan</b>.
    You can also open the <b>Case library</b> tab and click any row to analyse it.</div>
</div>
"""


def header_html():
    ds_ok = os.path.isdir(DATASET_ROOT)
    status = checkpoint_status()
    found = sum(status.values())
    return f"""
    <div class="pacs-header">
        <div>
            <div class="pacs-logo">KneeScan</div>
            <div class="pacs-sub">Knee MRI screening support for ACL tears, meniscal tears and joint abnormality</div>
        </div>
        <div class="chips">
            <span class="chip">{DEVICE.type.upper()}</span>
            <span class="chip {'chip-ok' if ds_ok else 'chip-bad'}">Dataset {'found' if ds_ok else 'not found'}</span>
            <span class="chip {'chip-ok' if found else 'chip-bad'}">{found}/{len(status)} checkpoints</span>
            <span class="chip chip-proxy">Research prototype</span>
        </div>
    </div>
    """


def model_card_html():
    status = checkpoint_status()
    rows = ''.join(
        f'<tr><td>{escape(k)}</td><td>{escape(CKPT[tuple(k.split(" / "))])}</td>'
        f'<td><span class="chip {"chip-ok" if ok else "chip-bad"}">{"loaded on demand" if ok else "missing"}</span></td></tr>'
        for k, ok in status.items()
    )
    c = MODEL_CARD
    return f"""
    <div class="report-container card-doc">
        <div class="findings-title">Architecture</div>
        <ul class="findings-list">
            <li><b>Encoder:</b> {escape(c['backbone'])}</li>
            <li><b>Sequence head:</b> {escape(c['head'])}</li>
            <li><b>Loss:</b> {escape(c['loss'])}</li>
            <li><b>Data:</b> {escape(c['data'])}</li>
        </ul>

        <div class="findings-title" style="margin-top:18px;">Reported performance</div>
        <table class="doc-table">
            <tr><th>Task</th><th>Baseline (max-pool)</th><th>BiLSTM + attention</th></tr>
            <tr><td>{escape(c['metric_task'])}</td><td>ROC-AUC {c['auc_baseline']:.4f}</td><td>ROC-AUC {c['auc_model']:.4f}</td></tr>
        </table>

        <div class="findings-title" style="margin-top:18px;">What each score means</div>
        <table class="doc-table">
            <tr><th>Card</th><th>How it is produced</th></tr>
            <tr><td>ACL</td><td>Trained BiLSTM per plane; probabilities fused with a logistic model when both planes are present.</td></tr>
            <tr><td>Meniscus</td><td>Dedicated checkpoint if present, otherwise a weighted proxy of the ACL probabilities.</td></tr>
            <tr><td>Overall abnormality</td><td>Dedicated checkpoint if present, otherwise a proxy derived from the other two cards.</td></tr>
        </table>

        <div class="findings-title" style="margin-top:18px;">Checkpoints</div>
        <table class="doc-table">
            <tr><th>Target / plane</th><th>File</th><th>Status</th></tr>{rows}
        </table>

        <div class="findings-title" style="margin-top:18px;">Limitations</div>
        <ul class="findings-list">
            <li>Validated on a single public dataset from one institution; no external validation.</li>
            <li>Grad-CAM highlights where the network responded. It is not a lesion segmentation.</li>
            <li>Probabilities are uncalibrated; the threshold slider changes the screening trade-off, not the model.</li>
            <li>Not a medical device. Do not use for diagnosis or treatment decisions.</li>
        </ul>
    </div>
    """


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------
def analyze(case_id, file_obj, upload_plane, threshold, history,
            level, window, show_cam, opacity, progress=gr.Progress()):
    t0 = time.time()
    progress(0.05, desc='Loading series')

    volumes = {}
    if file_obj is not None:
        path = getattr(file_obj, 'name', None) or str(file_obj)
        custom, case_str = True, os.path.basename(path)
        volumes[upload_plane] = load_volume(path)
    else:
        custom, case_str = False, str(case_id).zfill(4)
        for plane in PLANES:
            p = os.path.join(DATASET_ROOT, 'valid', plane, f'{case_str}.npy')
            if os.path.exists(p):
                volumes[plane] = load_volume(p)
    if not volumes:
        raise gr.Error(f'No series found for case {case_str}. Check the dataset path ({DATASET_ROOT}).')

    progress(0.25, desc='Running inference')
    probs, tensors, indices = {}, {}, {}
    for plane, vol in volumes.items():
        model = get_model('acl', plane)
        if model is None:
            continue
        tensors[plane], indices[plane] = to_tensor(vol, plane)
        probs[plane] = predict(model, tensors[plane])
    if not probs:
        expected = ', '.join(CKPT[('acl', p)] for p in volumes)
        raise gr.Error(f'No trained checkpoint found for the available series. Expected: {expected}')
    unused = [p for p in volumes if p not in probs]
    scores = compute_scores(probs, tensors)

    progress(0.65, desc='Computing Grad-CAM')
    cams = {}
    for plane in probs:
        try:
            cams[plane] = run_gradcam(plane, get_model('acl', plane), tensors[plane], indices[plane])
        except Exception:
            pass  # analysis still succeeds without a heatmap

    ranges = {}
    for plane, vol in volumes.items():
        lo, hi = np.percentile(vol, (0.5, 99.5))
        if hi <= lo:
            lo, hi = float(vol.min()), float(vol.max()) + 1.0
        ranges[plane] = (float(lo), float(hi))

    labels = get_labels()
    gt = {t: (labels[t].get(case_str) if not custom else None) for t in TASKS}

    ctx = dict(
        case=case_str, custom=custom, volumes=volumes, indices=indices, cams=cams, probs=probs,
        scores=scores, gt=gt, range=ranges, unused_planes=unused,
        primary='axial' if 'axial' in probs else next(iter(probs)),
        ts=dt.datetime.now().strftime('%H:%M:%S'), latency_ms=(time.time() - t0) * 1000.0,
    )

    progress(0.9, desc='Rendering')
    slider_u, img, strip = viewer_bundle(ctx, ctx['primary'], level, window, show_cam, opacity)

    gt_acl = ctx['gt'].get('acl')
    row = [
        ctx['ts'], 'upload' if custom else case_str,
        round(scores['acl']['prob'] * 100, 1), round(scores['meniscus']['prob'] * 100, 1),
        round(scores['abnormal']['prob'] * 100, 1),
        '-' if gt_acl is None else ('Positive' if gt_acl == 1 else 'Negative'),
        round(ctx['latency_ms']),
    ]
    history = (history or []) + [row]

    planes_available = [(PLANE_LABEL[p], p) for p in PLANES if p in volumes]
    return (
        ctx,
        render_report(ctx, threshold),
        gr.update(choices=planes_available, value=ctx['primary']),
        slider_u, img, strip,
        history, history,
        gr.update(visible=False, value=None),
    )


def rerender_report(ctx, thr):
    return render_report(ctx, thr)


def on_view_change(ctx, plane, idx, level, window, show_cam, opacity):
    if not ctx or plane not in ctx['volumes']:
        return None
    return render_core(ctx, plane, int(idx) - 1, level, window, show_cam, opacity)


def on_plane_change(ctx, plane, level, window, show_cam, opacity):
    if not ctx or plane not in ctx['volumes']:
        return gr.update(), None, []
    return viewer_bundle(ctx, plane, level, window, show_cam, opacity)


def jump_to_key(ctx, plane):
    cam = ctx['cams'].get(plane) if ctx else None
    return int(cam['raw_idx']) + 1 if cam else gr.update()


def step_slice(cur, delta, ctx, plane):
    if not ctx or plane not in ctx['volumes']:
        return gr.update()
    return int(min(max(int(cur) + delta, 1), ctx['volumes'][plane].shape[0]))


def jump_from_filmstrip(evt: gr.SelectData, ctx, plane):
    if not ctx or plane not in ctx['indices']:
        return gr.update()
    pos = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    return int(ctx['indices'][plane][int(pos)]) + 1


def reset_window():
    return 0.5, 1.0


def random_case():
    return random.choice(list_case_ids()), None


def library_rows(filter_name='All'):
    labels = get_labels()

    def mark(task, cid):
        v = labels[task].get(cid)
        return '-' if v is None else ('Positive' if v == 1 else 'Negative')

    rows = []
    for cid in list_case_ids():
        row = [cid, mark('acl', cid), mark('meniscus', cid), mark('abnormal', cid)]
        keep = {
            'All': True,
            'ACL tear': row[1] == 'Positive',
            'No ACL tear': row[1] == 'Negative',
            'Meniscus tear': row[2] == 'Positive',
            'Abnormal': row[3] == 'Positive',
        }[filter_name]
        if keep:
            rows.append(row)
    return rows


def pick_from_library(evt: gr.SelectData, table):
    row_value = getattr(evt, 'row_value', None)
    if row_value:
        case = str(row_value[0])
    else:
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        case = str(table.iloc[row, 0]) if hasattr(table, 'iloc') else str(table[row][0])
    return case, None, gr.update(selected='analysis')


def export_report(ctx, thr):
    if not ctx:
        raise gr.Error('Run an analysis first, then export the report.')
    plane = ctx['primary']
    cam = ctx['cams'].get(plane)
    n = ctx['volumes'][plane].shape[0]
    idx = cam['raw_idx'] if cam else n // 2

    def b64(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()

    raw = render_core(ctx, plane, idx, 0.5, 1.0, False, 0.0)
    over = render_core(ctx, plane, idx, 0.5, 1.0, True, 0.45)
    figures = f"""
    <div class="report-container" style="margin-top:16px;">
        <div class="findings-title">Key slice ({PLANE_LABEL[plane]}, slice {idx + 1} of {n})</div>
        <div style="display:flex; gap:16px; flex-wrap:wrap;">
            <img src="data:image/png;base64,{b64(raw)}" style="max-width:48%; border-radius:8px;"/>
            <img src="data:image/png;base64,{b64(over)}" style="max-width:48%; border-radius:8px;"/>
        </div>
    </div>
    """
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>KneeScan report</title>
    <style>{CUSTOM_CSS} body {{ padding: 28px; max-width: 1100px; margin: 0 auto; }}</style></head>
    <body>{header_html()}{render_report(ctx, thr)}{figures}
    <p class="muted" style="margin-top:18px;">Generated {dt.datetime.now():%Y-%m-%d %H:%M} by KneeScan. Research prototype, not for clinical use.</p>
    </body></html>"""

    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', ctx['case'])
    path = os.path.join(tempfile.mkdtemp(), f'kneescan_report_{safe}_{dt.datetime.now():%H%M%S}.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    return gr.update(value=path, visible=True)


def export_history(history):
    if not history:
        raise gr.Error('Nothing to export yet.')
    path = os.path.join(tempfile.mkdtemp(), f'kneescan_session_{dt.datetime.now():%Y%m%d_%H%M%S}.csv')
    pd.DataFrame(history, columns=HIST_COLS).to_csv(path, index=False)
    return gr.update(value=path, visible=True)


def clear_history():
    return [], [], gr.update(visible=False, value=None)


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    color: #e2e8f0 !important;
}
.gradio-container { max-width: 1500px !important; }

.pacs-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 12px; padding: 16px 22px; margin-bottom: 16px;
    display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.pacs-logo { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.5px; color: #38bdf8; }
.pacs-sub { font-size: 0.85rem; color: #94a3b8; margin-top: 2px; }

.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
    display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 20px;
    background: #1e293b; border: 1px solid #334155; color: #cbd5e1;
}
.chip-ok { background: rgba(16,185,129,.15); border-color: rgba(16,185,129,.4); color: #6ee7b7; }
.chip-bad { background: rgba(239,68,68,.15); border-color: rgba(239,68,68,.4); color: #fca5a5; }
.chip-proxy { background: rgba(245,158,11,.15); border-color: rgba(245,158,11,.45); color: #fcd34d; }
.chip-src { background: rgba(56,189,248,.12); border-color: rgba(56,189,248,.35); color: #7dd3fc; }

.block, .panel { background: #111827 !important; border: 1px solid #1f2937 !important; border-radius: 12px !important; }

.btn-primary {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important; border: none !important;
    color: #ffffff !important; font-weight: 700 !important; font-size: 1.02rem !important;
    border-radius: 10px !important; box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s ease !important;
}
.btn-primary:hover { background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%) !important; transform: translateY(-1px); }

.report-container { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 20px; }
.report-header {
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid #1f2937; padding-bottom: 14px; margin-bottom: 18px;
}
.exam-title { font-size: 1.15rem; font-weight: 700; color: #f8fafc; display: block; }
.exam-meta { font-size: 0.82rem; color: #94a3b8; }
.empty-state { text-align: center; color: #64748b; padding: 46px 20px; }
.empty-title { font-size: 1.1rem; font-weight: 600; color: #94a3b8; margin-bottom: 8px; }
.muted { color: #64748b; }

.targets-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }
.target-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px;
    display: flex; flex-direction: column; justify-content: space-between;
}
.target-card-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 10px; }
.target-title { font-size: 0.98rem; font-weight: 700; color: #f8fafc; }
.risk-display { margin: 6px 0; }
.risk-value { font-size: 1.9rem; font-weight: 800; display: block; line-height: 1.1; }
.risk-label { font-size: 0.74rem; color: #94a3b8; }

.progress-bar-bg { position: relative; height: 8px; background: #0f172a; border-radius: 4px; margin: 12px 0; }
.progress-bar-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
.thr-tick { position: absolute; top: -3px; bottom: -3px; width: 2px; background: #e2e8f0; border-radius: 1px; }

.meta-line { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }
.meta-key { display: inline-block; min-width: 82px; color: #64748b; }
.target-card-footer { margin-top: 8px; }

.status-badge {
    display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 20px;
    font-size: 0.74rem; font-weight: 700; white-space: nowrap;
}
.badge-pos { background: rgba(239,68,68,.18); border: 1px solid rgba(239,68,68,.4); color: #fca5a5; }
.badge-neg { background: rgba(16,185,129,.18); border: 1px solid rgba(16,185,129,.4); color: #6ee7b7; }
.badge-border { background: rgba(245,158,11,.18); border: 1px solid rgba(245,158,11,.45); color: #fcd34d; }
.badge-dot { width: 6px; height: 6px; border-radius: 50%; background-color: currentColor; }

.findings-box { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 14px; margin-top: 16px; }
.findings-title { font-size: 0.9rem; font-weight: 700; color: #38bdf8; margin-bottom: 8px; }
.findings-list { margin: 0; padding-left: 18px; font-size: 0.85rem; color: #cbd5e1; line-height: 1.65; }

.doc-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.doc-table th, .doc-table td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #1f2937; color: #cbd5e1; }
.doc-table th { color: #94a3b8; font-weight: 600; }
"""

THEME = gr.themes.Monochrome(primary_hue='sky', neutral_hue='slate')


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def build_app():
    case_ids = list_case_ids()

    with gr.Blocks(css=CUSTOM_CSS, theme=THEME, title='KneeScan - knee MRI analysis') as demo:
        ctx_state = gr.State(None)
        hist_state = gr.State([])

        gr.HTML(header_html())

        with gr.Tabs() as tabs:
            # ------------------------------------------------------------ Analysis
            with gr.TabItem('Analysis', id='analysis'):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown('### Scan')
                        case_dd = gr.Dropdown(choices=case_ids, value=case_ids[0], filterable=True,
                                              label='Validation case (Stanford MRNet)')
                        random_btn = gr.Button('Random case', size='sm')
                        file_up = gr.File(label='Or upload a .npy volume (slices, H, W)',
                                          file_types=['.npy'], interactive=True)
                        upload_plane = gr.Radio([('Axial PD', 'axial'), ('Sagittal T2', 'sagittal')],
                                                value='axial', label='Plane of the uploaded volume')
                        gr.Markdown('### Screening threshold')
                        thr = gr.Slider(0.05, 0.95, value=0.50, step=0.01,
                                        label='Decision threshold (probability)',
                                        info='Lower catches more tears but raises false alarms. Updates instantly.')
                        analyze_btn = gr.Button('Analyse scan', elem_classes=['btn-primary'], size='lg')
                        gr.Markdown('An uploaded file overrides the case selection.')

                    with gr.Column(scale=3):
                        report = gr.HTML(EMPTY_REPORT)

                        gr.Markdown('### Slice viewer')
                        with gr.Row():
                            with gr.Column(scale=2):
                                viewer = gr.Image(label='Viewer', show_label=False, type='numpy',
                                                  interactive=False, height=520)
                                with gr.Row():
                                    prev_btn = gr.Button('Previous', size='sm')
                                    slice_sl = gr.Slider(1, 36, value=1, step=1, label='Slice')
                                    next_btn = gr.Button('Next', size='sm')
                            with gr.Column(scale=1, min_width=260):
                                plane_rd = gr.Radio([('Axial PD', 'axial'), ('Sagittal T2', 'sagittal')],
                                                    value='axial', label='Series')
                                level_sl = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label='Window level')
                                window_sl = gr.Slider(0.05, 1.0, value=1.0, step=0.01, label='Window width')
                                reset_btn = gr.Button('Reset window', size='sm')
                                cam_cb = gr.Checkbox(value=True, label='Grad-CAM overlay (key slice only)')
                                opacity_sl = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label='Overlay opacity')
                                key_btn = gr.Button('Jump to key slice', size='sm')
                        filmstrip = gr.Gallery(label='Slices the model saw (click to open)', columns=8, rows=2,
                                               height=250, object_fit='cover', allow_preview=False)

                        with gr.Row():
                            export_btn = gr.Button('Export report (HTML)', size='sm')
                            export_file = gr.File(label='Report', visible=False, interactive=False)

            # ------------------------------------------------------------ Case library
            with gr.TabItem('Case library', id='library'):
                gr.Markdown('Click any row to open that case in the Analysis tab and run it.')
                lib_filter = gr.Radio(['All', 'ACL tear', 'No ACL tear', 'Meniscus tear', 'Abnormal'],
                                      value='All', label='Filter by ground truth')
                lib_table = gr.Dataframe(value=library_rows('All'),
                                         headers=['Case', 'ACL', 'Meniscus', 'Abnormal'],
                                         interactive=False, wrap=False)

            # ------------------------------------------------------------ History
            with gr.TabItem('Session history', id='history'):
                hist_df = gr.Dataframe(value=[], headers=HIST_COLS, interactive=False, wrap=True)
                with gr.Row():
                    hist_export = gr.Button('Export CSV', size='sm')
                    hist_clear = gr.Button('Clear history', size='sm')
                hist_file = gr.File(label='Session export', visible=False, interactive=False)

            # ------------------------------------------------------------ Model card
            with gr.TabItem('Model card', id='model'):
                gr.HTML(model_card_html())

        # ------------------------------------------------------------ Wiring
        view_inputs = [ctx_state, plane_rd, slice_sl, level_sl, window_sl, cam_cb, opacity_sl]
        plane_inputs = [ctx_state, plane_rd, level_sl, window_sl, cam_cb, opacity_sl]

        analyze_inputs = [case_dd, file_up, upload_plane, thr, hist_state,
                          level_sl, window_sl, cam_cb, opacity_sl]
        analyze_outputs = [ctx_state, report, plane_rd, slice_sl, viewer, filmstrip,
                           hist_state, hist_df, export_file]

        analyze_btn.click(analyze, analyze_inputs, analyze_outputs)

        random_btn.click(random_case, None, [case_dd, file_up]).then(analyze, analyze_inputs, analyze_outputs)

        thr.change(rerender_report, [ctx_state, thr], report, show_progress='hidden')

        for ctrl in (slice_sl, level_sl, window_sl, cam_cb, opacity_sl):
            ctrl.change(on_view_change, view_inputs, viewer, show_progress='hidden', trigger_mode='always_last')

        plane_rd.change(on_plane_change, plane_inputs, [slice_sl, viewer, filmstrip], show_progress='hidden')
        key_btn.click(jump_to_key, [ctx_state, plane_rd], slice_sl)
        prev_btn.click(lambda cur, c, p: step_slice(cur, -1, c, p), [slice_sl, ctx_state, plane_rd], slice_sl)
        next_btn.click(lambda cur, c, p: step_slice(cur, 1, c, p), [slice_sl, ctx_state, plane_rd], slice_sl)
        filmstrip.select(jump_from_filmstrip, [ctx_state, plane_rd], slice_sl)
        reset_btn.click(reset_window, None, [level_sl, window_sl])

        lib_filter.change(library_rows, lib_filter, lib_table)
        lib_table.select(pick_from_library, lib_table, [case_dd, file_up, tabs]).then(
            analyze, analyze_inputs, analyze_outputs)

        export_btn.click(export_report, [ctx_state, thr], export_file)
        hist_export.click(export_history, hist_state, hist_file)
        hist_clear.click(clear_history, None, [hist_state, hist_df, hist_file])

    return demo


if __name__ == '__main__':
    app = build_app()
    for port in [7865, 7866, 7867, 7868, 7860, 7861]:
        try:
            print(f'Starting KneeScan on http://127.0.0.1:{port} ...', flush=True)
            app.launch(server_name='127.0.0.1', server_port=port, share=False)
            break
        except OSError:
            continue