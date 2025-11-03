# ORBET – Detección + Clasificación con alertas SOLO TELEGRAM (SIN impresión)
# Tickets y capturas se guardan automáticamente en subcarpetas por fecha.
# Requisitos: pip install ultralytics opencv-python requests

import os, cv2, time, sqlite3, subprocess, sys, platform, requests, threading, queue
from collections import Counter
from datetime import datetime
from ultralytics import YOLO

# ========================= CONFIGURACIÓN =========================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR   = os.path.join(BASE_DIR, "capturas")
TICKETS_DIR   = os.path.join(BASE_DIR, "tickets")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")
DB_FILE       = os.path.join(BASE_DIR, "ingresos.db")
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(TICKETS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Fuente de video ---
CAM_SOURCE = 0
# CAM_SOURCE = "rtsp://admin:Ares12345@192.168.1.64:554/Streaming/Channels/401"

# --- Detección de movimiento ---
PREVIEW_BOXES   = True
AREA_MINIMA     = 1200
COOLDOWN_GLOBAL = 2.0
COOLDOWN_POR_CAT= 4.0

# --- YOLO ---
YOLO_WEIGHTS    = "yolov8n.pt"
YOLO_CONF       = 0.25
YOLO_IMGSZ      = 512
RESIZE_FOR_YOLO = True

# --- Tickets / impresión ---
# IMPORTANTE: Desactivamos impresión para evitar la ventana "Guardar como".
AUTO_PRINT             = False   # <<<< DESACTIVADO
IMPRIMIR_VIA_NOTEPAD   = False
PRINT_COOLDOWN_SECONDS = 10.0

# --- Telegram ---
ENABLE_TELEGRAM       = True
TELEGRAM_ONLY_PERSONA = True
TELEGRAM_TOKEN        = "8263890132:AAG2AK6NWmH41OprRz0gaMKyIcF4FA8jun0"
TELEGRAM_CHAT_ID      = "8334421914"

# --- Mapeos COCO -> grupos ---
VEHICLES = {"car","truck","bus","motorcycle","bicycle","train","boat"}
ANIMALS  = {"dog","cat","bird","horse","sheep","cow","elephant","bear","zebra","giraffe"}
PERSON   = {"person"}

# ========================= UTILIDADES =========================
def log_line(text: str):
    lf = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.txt")
    with open(lf, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def now_str(fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.now().strftime(fmt)

def open_camera(source):
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap

def _cols(conn, table):
    cur = conn.cursor(); cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def init_or_migrate_db():
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS ingresos (id INTEGER PRIMARY KEY AUTOINCREMENT)")
    conn.commit()
    desired = {
        "timestamp":"TEXT","image_path":"TEXT","ticket_path":"TEXT","note":"TEXT","ticket_type":"TEXT",
        "category":"TEXT","raw_class":"TEXT","confidence":"REAL","model_name":"TEXT","source":"TEXT","alert_sent":"INTEGER"
    }
    existing = _cols(conn, "ingresos")
    for col, typ in desired.items():
        if col not in existing:
            c.execute(f"ALTER TABLE ingresos ADD COLUMN {col} {typ}")
            conn.commit()
    conn.close()

def day_folder(base):
    d = os.path.join(base, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(d, exist_ok=True)
    return d

def save_capture(image, category):
    folder = day_folder(CAPTURE_DIR)
    ts = datetime.now().strftime("%H-%M-%S")
    fname = f"{category or 'evento'}_{ts}.jpg"
    path = os.path.join(folder, fname)
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return path

def create_ticket_entry(image_path, category, raw_class, conf):
    folder = day_folder(TICKETS_DIR)  # => tickets/AAAA-MM-DD/
    ts_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ticket_path = os.path.join(folder, f"ticket_{ts_id}.txt")
    with open(ticket_path, "w", encoding="utf-8") as f:
        f.write("TICKET DE REGISTRO - ORBET\n")
        f.write("==========================\n")
        f.write(f"ID: {ts_id}\n")
        f.write(f"FECHA_HORA: {now_str()}\n")
        f.write(f"IMAGEN: {image_path}\n")
        f.write(f"CATEGORÍA: {category or 'DESCONOCIDO'}\n")
        if raw_class is not None:
            f.write(f"CLASE_YOLO: {raw_class} (conf={conf:.2f})\n")
        f.write("\nCATEGORÍA (marcar con X):\n")
        f.write("[ ] MOVIL   [ ] TRABAJADOR   [ ] VISITA   [ ] CLIENTE\n")
        f.write("\nFirma vigilante: _________________________________\n")
    return ticket_path

def db_insert(**kwargs):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute(
        """
        INSERT INTO ingresos
        (timestamp,image_path,ticket_path,note,ticket_type,category,raw_class,confidence,model_name,source,alert_sent)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            kwargs.get("timestamp"),
            kwargs.get("image_path"),
            kwargs.get("ticket_path"),
            kwargs.get("note","Evento detectado - ticket creado"),
            kwargs.get("ticket_type"),
            kwargs.get("category"),
            kwargs.get("raw_class"),
            kwargs.get("confidence"),
            kwargs.get("model_name"),
            kwargs.get("source"),
            1 if kwargs.get("alert_sent") else 0
        )
    )
    conn.commit(); conn.close()

# ========================= WORKER ASÍNCRONO (Telegram) =========================
ALERT_QUEUE = queue.Queue()
_last_print_time = 0.0  # reservado por si vuelves a activar impresión

def alert_worker():
    session = requests.Session()
    while True:
        item = ALERT_QUEUE.get()
        if item is None:
            break
        try:
            if item["type"] == "telegram":
                try:
                    with open(item["image_path"], "rb") as fh:
                        session.post(
                            f"https://api.telegram.org/bot{item['token']}/sendPhoto",
                            data={"chat_id": item["chat_id"], "caption": item["caption"]},
                            files={"photo": fh},
                            timeout=8,
                        )
                except Exception:
                    session.post(
                        f"https://api.telegram.org/bot{item['token']}/sendMessage",
                        data={"chat_id": item["chat_id"], "text": item["caption"]},
                        timeout=5,
                    )
            # NOTA: ya no hay rama "print" para evitar abrir Bloc de notas.
        finally:
            ALERT_QUEUE.task_done()

ALERT_THREAD = threading.Thread(target=alert_worker, daemon=True)
ALERT_THREAD.start()

# ========================= CLASIFICACIÓN / YOLO =========================
def map_label_to_group(cls_name: str):
    n = cls_name.lower()
    if n in PERSON:   return "PERSONA"
    if n in VEHICLES: return "AUTO"
    if n in ANIMALS:  return "ANIMAL"
    return None

def run_yolo(model: YOLO, frame, conf_thr=0.35, imgsz=640):
    results = model.predict(frame, imgsz=imgsz, conf=conf_thr, stream=False, verbose=False)
    best = (None, None, 0.0, None)
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            c = int(box.cls[0])
            cls_name = model.names[c] if hasattr(model, "names") else str(c)
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            group = map_label_to_group(cls_name)
            if group is None:
                continue
            if conf > best[2]:
                xyxy = tuple(map(int, box.xyxy[0].tolist()))
                best = (group, cls_name, conf, xyxy)
    return best

# ========================= LOOP PRINCIPAL =========================
def main():
    print(f"📂 BD: {DB_FILE}")
    init_or_migrate_db()

    print("🧠 Cargando modelo YOLO…")
    model = YOLO(YOLO_WEIGHTS)

    # Warm-up
    try:
        import numpy as np
        dummy = (np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8))
        model.predict(dummy, imgsz=YOLO_IMGSZ, conf=0.1, verbose=False)
    except Exception:
        pass

    cap = open_camera(CAM_SOURCE)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la fuente de video.")
        return

    print("✅ ORBET monitoreando… (q para salir)")

    last_gray = None
    last_event_ts = 0.0
    last_cat_ts = {"PERSONA": 0.0, "ANIMAL": 0.0, "AUTO": 0.0}
    counters = Counter()

    t_prev = time.time(); fps = 0.0

    while True:
        try:
            for _ in range(2):
                cap.grab()
        except Exception:
            pass

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        full_frame = frame.copy()

        # FPS
        t_now = time.time(); dt = t_now - t_prev
        if dt > 0: fps = 0.9*fps + 0.1*(1.0/dt)
        t_prev = t_now

        # Movimiento
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if last_gray is None:
            last_gray = gray
            cv2.putText(frame, now_str(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow("ORBET Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        frame_delta = cv2.absdiff(last_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hay_movimiento = any(cv2.contourArea(c) >= AREA_MINIMA for c in contours)
        if PREVIEW_BOXES:
            for c in contours:
                if cv2.contourArea(c) >= AREA_MINIMA:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # HUD
        cv2.putText(frame, now_str(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Eventos: P={counters['PERSONA']} A={counters['ANIMAL']} V={counters['AUTO']}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("ORBET Monitor", frame)

        # Trigger YOLO
        if hay_movimiento and (time.time() - last_event_ts >= COOLDOWN_GLOBAL):
            last_event_ts = time.time()

            infer_frame = full_frame
            scale_x = scale_y = 1.0
            if RESIZE_FOR_YOLO:
                h, w = full_frame.shape[:2]
                r = YOLO_IMGSZ / max(h, w)
                new_w, new_h = int(w * r), int(h * r)
                infer_frame = cv2.resize(full_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scale_x, scale_y = w / new_w, h / new_h

            category, raw_class, conf, xyxy = run_yolo(model, infer_frame, YOLO_CONF, YOLO_IMGSZ)

            if category is not None:
                if time.time() - last_cat_ts.get(category, 0.0) < COOLDOWN_POR_CAT:
                    last_gray = gray
                    continue
                last_cat_ts[category] = time.time()

                if xyxy and RESIZE_FOR_YOLO:
                    x1, y1, x2, y2 = xyxy
                    xyxy = (int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y))

                evid = full_frame.copy()
                if xyxy:
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(evid, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(evid, f"{category} ({conf:.2f})", (x1, max(20, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                # Guardado automático (captura + ticket)
                image_path  = save_capture(evid, category)
                ticket_path = create_ticket_entry(image_path, category, raw_class, conf)
                ts_now = now_str()

                # BD
                db_insert(
                    timestamp=ts_now,
                    image_path=image_path,
                    ticket_path=ticket_path,
                    note=f"Movimiento + YOLO: {category}",
                    ticket_type=None,
                    category=category,
                    raw_class=raw_class,
                    confidence=float(conf),
                    model_name=YOLO_WEIGHTS,
                    source=str(CAM_SOURCE),
                    alert_sent=1
                )

                counters[category] += 1
                log_line(f"[{ts_now}] {category} detectada – {raw_class} (conf={conf:.2f}) → {os.path.basename(image_path)}")

                # Telegram (opcional)
                if ENABLE_TELEGRAM and (not TELEGRAM_ONLY_PERSONA or category == "PERSONA"):
                    tcap = f"🚨 ORBET: {category} – {raw_class} (conf={conf:.2f})\n{ts_now}"
                    ALERT_QUEUE.put({
                        "type":"telegram",
                        "image_path": image_path,
                        "caption": tcap,
                        "token": TELEGRAM_TOKEN,
                        "chat_id": TELEGRAM_CHAT_ID
                    })

        last_gray = gray
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()
    try:
        ALERT_QUEUE.put(None)
    except Exception:
        pass

if __name__ == "__main__":
    main()
