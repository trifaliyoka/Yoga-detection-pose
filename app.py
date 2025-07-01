from flask import Flask, render_template, request
import re, os, logging, cv2, mediapipe as mp
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
from markupsafe import Markup
from pose_analysis import analyze_video

# ───── konfigurasi ─────
app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER   = 'static/videos',
    FRAME_FOLDER    = 'static/frames',
    OVERLAY_FOLDER  = 'static/videos_overlay',
    ALLOWED_EXTENSIONS = {'mp4', 'webm', 'ogg'}
)
for f in (app.config['UPLOAD_FOLDER'],
          app.config['FRAME_FOLDER'],
          app.config['OVERLAY_FOLDER']):
    os.makedirs(f, exist_ok=True)

CONF_THRESH = 0.80

# ───── logging ─────
handler = RotatingFileHandler('app.log', maxBytes=1_000_000, backupCount=3)
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S'))
app.logger.setLevel(logging.INFO)
app.logger.addHandler(handler)

# ───── filter jinja2 ─────
@app.template_filter("body_id")
def body_id_filter(text:str)->Markup:
    mapping = {"left_elbow":"siku kiri","right_elbow":"siku kanan",
               "left_shoulder":"bahu kiri","right_shoulder":"bahu kanan",
               "left_hip":"pinggul kiri","right_hip":"pinggul kanan",
               "left_knee":"lutut kiri","right_knee":"lutut kanan",
               "left_wrist":"pergelangan kiri","right_wrist":"pergelangan kanan"}
    for en,idn in mapping.items():
        text = re.sub(rf"\b{en.replace('_',' ').title()}\b", idn, text, flags=re.I)
    return Markup(text)

# ───── util ─────
def allowed(fname:str)->bool:
    return '.' in fname and fname.rsplit('.',1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_conf(pred:str)->float:
    m = re.search(r'\(([\d.]+)%\)', pred or '')
    return float(m.group(1))/100 if m else 0.0

# ───── overlay key‑point ─────
def overlay_keypoint(src:str, dst:str)->bool:
    """Kembalikan True jika overlay berhasil dibuat."""
    mp_pose, mp_draw = mp.solutions.pose, mp.solutions.drawing_utils
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        app.logger.error("VideoCapture gagal membuka %s", src)
        return False

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0 or h == 0:
        app.logger.error("Dimensi video 0×0 – batal overlay")
        cap.release(); return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*'avc1')      # ← gunakan H.264
    out = cv2.VideoWriter(dst, fourcc, fps, (w, h))

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(0,255,0), thickness=2, circle_radius=3))
            out.write(frame)

    cap.release(); out.release()
    app.logger.info("Overlay selesai tulis %s (%dx%d)", dst, w, h)
    return True

# ───── routes ─────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    ctx = dict(filename=None, frame_image=None,
               prediction=None, feedback_list=[])

    file = request.files.get('video')
    if file and allowed(file.filename):
        raw_name  = secure_filename(file.filename)
        src_path  = os.path.join(app.config['UPLOAD_FOLDER'], raw_name)
        file.save(src_path)
        app.logger.info('Upload diterima: %s', raw_name)

        # coba buat overlay
        overlay_name = f"overlay_{raw_name}"
        overlay_path = os.path.join(app.config['OVERLAY_FOLDER'], overlay_name)
        success      = overlay_keypoint(src_path, overlay_path)

        # frame dan analisis
        frame_name = f"{os.path.splitext(raw_name)[0]}_frame.jpg"
        frame_path = os.path.join(app.config['FRAME_FOLDER'], frame_name)
        pred, fb   = analyze_video(src_path, frame_path)
        conf       = extract_conf(pred)
        app.logger.info('Prediksi: %s | confidence=%.2f', pred, conf)

        if pred is None or conf < CONF_THRESH:
            app.logger.warning('Confidence < threshold – pose ditolak')
            pred, fb = "Pose tidak terdeteksi", []

        # pakai overlay jika sukses, kalau gagal tampilkan video asli
        final_name = overlay_name if success else raw_name
        if not success:
            app.logger.warning("Overlay gagal – tampilkan video asli")

        ctx.update(filename=final_name,
                   frame_image=frame_name,
                   prediction=pred,
                   feedback_list=fb)

    elif request.method == 'POST':
        app.logger.warning('Tidak ada file valid yang di-upload')

    return render_template('predict.html', **ctx)

# ───── main ─────
if __name__ == '__main__':
    app.run(debug=True)
