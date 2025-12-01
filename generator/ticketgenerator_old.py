#!/usr/bin/env python3
"""
Generate synthetic IT Incidents (Tickets) with a local Ollama Llama model,
cluster them heuristically, and generate Knowledge Base articles with a Qwen model.

- Tickets: llama3.1:8b-instruct
- KB:      qwen2.5:7b-instruct

CSV-Schema für Incidents passt zu deinem bisherigen Setup:

Incidents: synthetic_incidents_llm.csv
  - ticket_id
  - title
  - description
  - created_at
  - impact
  - urgency
  - priority_level
  - priority
  - status
  - category
  - service
  - category_path
  - ci_id
  - os
  - hostname
  - reporter
  - assigned_group
  - assignee
  - site
  - conversation_history
  - comments_count
  - error_code
  - gold_kb_id       (aktuell leer)
  - gold_resolution  (aktuell leer)
  - issue_type
  - ticket_fulltext

KB: synthetic_kb_llm.csv
  - kb_id
  - title
  - category
  - service
  - problem
  - symptoms
  - root_cause
  - troubleshooting_steps
  - resolution_steps
  - references
  - kb_fulltext
  - os
  - tags
  - related_error_codes

Logging:
  - generation_log.csv mit Dauer und Textlänge für jeden Ticket- und KB-Call
  - Konsolenausgabe + Gesamtzeit
"""

import os
import csv
import json
import time
import random

from datetime import datetime, timedelta
from collections import defaultdict
from bin.config import OllamaConfig
from bin.text_utils import safe_split, safe_parse_level
from bin.logging_utils import logging
from bin.metrics_utils import ollama_metrics
from benchmark.benchmark import extract_ollama_metrics

import pandas as pd
import requests

# ------------------ CONFIG ------------------

OLLAMA_URL = OllamaConfig().url
OLLAMA_MODEL = OllamaConfig().model
#TICKET_MODEL = "llama3.1:8b-instruct"
TICKET_MODEL = OLLAMA_MODEL
KB_MODEL = "qwen2.5:7b-instruct"

# Pfase definieren
OUT_DIR = "./output"
LOG_FILE = "generation_log.csv"

# Ausgabedateien
INC_CSV_FILE = "synthetic_incidents_llm_text10.csv"
KB_CSV_FILE = "synthetic_kb_llm_test1ß.csv"

# Gesamtzahl der zu generierenden Tickets
TOTAL_TICKETS = int(os.getenv("TOTAL_TICKETS", "10000"))

# Wie viele Tickets pro Ollama-Call erzeugt werden sollen
TICKETS_PER_CALL = int(os.getenv("TICKETS_PER_CALL", "10"))

# wie viele Tickets pro KB-Kontext (Prompt)
MAX_TICKETS_PER_KB_CONTEXT = 15  

SEED = 1234

random.seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# Logging Setup
LOG_CSV_PATH = os.path.join(OUT_DIR, LOG_FILE)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def sanitize_model_name(model: str) -> str:
    """
    Wandelt Modellnamen in etwas um, das gut als Dateiname funktioniert.
    Beispiel: "llama3.1:8b-instruct" -> "llama3_1_8b-instruct"
    """
    bad_chars = [":", "/", "\\", " "]
    safe = model
    for ch in bad_chars:
        safe = safe.replace(ch, "_")
    return safe



def init_log_csv():
    if not os.path.exists(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "phase",       # "ticket" oder "kb"
                    "key",         # index oder gruppenkey
                    "duration_sec",
                    "response_chars",
                    "ok",
                    "error"
                ]
            )
            w.writeheader()

def log_call(phase: str, key: str, duration: float, text_len: int, ok: bool, error: str = ""):
    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["phase","key","duration_sec","response_chars","ok","error"]
        )
        w.writerow({
            "phase": phase,
            "key": key,
            "duration_sec": round(duration, 3),
            "response_chars": text_len,
            "ok": int(ok),
            "error": error
        })

# ------------------ Helper-Funktionen ------------------

def sn_inc_id(n: int) -> str:
    return f"INC{n:08d}"

def sn_kb_id(n: int) -> str:
    return f"KB{n:07d}"

def sn_ci_id(n: int) -> str:
    return f"CI{n:08d}"

def fmt_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat() + "Z"

def random_created_at(days_back: int = 180) -> datetime:
    return datetime.utcnow() - timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

def make_category_path(category: str, service: str) -> str:
    return f"IT Services > {category} > {service}"

FIRST_NAMES = ["Anna","Ben","Chris","Dana","Erik","Fatma","Lukas","Mara","Noah","Olga","Paul","Rita"]
LAST_NAMES  = ["Meyer","Schmidt","Schneider","Fischer","Weber","Wagner","Becker","Hoffmann","Klein","Wolf"]
SITES       = ["Berlin","München","Hamburg","Köln","Remote"]
GROUPS      = ["1st Level Support","2nd Level Netzwerk","Workplace","Datenbank-Team"]

def rand_name() -> str:
    return random.choice(FIRST_NAMES) + " " + random.choice(LAST_NAMES)

def rand_hostname() -> str:
    prefix = random.choice(["ws","nb","srv"])
    return f"{prefix}-{random.randint(1000,9999)}.corp.local"

OSES = ["Windows 10","Windows 11"]

CATEGORIES_SERVICES = [
    # --- Network ---
    ("Network", "VPN"),
    ("Network", "DNS"),
    ("Network", "DHCP"),
    ("Network", "Firewall"),
    ("Network", "Proxy"),
    ("Network", "Load Balancer"),
    ("Network", "WLAN"),
    ("Network", "Switching"),
    ("Network", "Routing"),
    ("Network", "TLS/SSL"),

    # --- Access & Identity ---
    ("Access", "AD Login"),
    ("Access", "LDAP"),
    ("Access", "SSO"),
    ("Access", "MFA"),
    ("Access", "Password Reset"),
    ("Access", "Kerberos"),
    ("Access", "Azure AD"),
    ("Access", "Conditional Access"),

    # --- Hardware / Workplace ---
    ("Hardware", "Printer"),
    ("Hardware", "Scanner"),
    ("Hardware", "ThinClient"),
    ("Hardware", "Monitor"),
    ("Hardware", "Dockingstation"),
    ("Hardware", "Webcam"),
    ("Hardware", "Keyboard"),
    ("Hardware", "Headset"),

    # --- Software / Desktop / Client ---
    ("Software", "Office"),
    ("Software", "Outlook"),
    ("Software", "Teams"),
    ("Software", "Browser"),
    ("Software", "PDF Viewer"),
    ("Software", "Antivirus Client"),
    ("Software", "VPN Client"),
    ("Software", "Java Runtime"),
    ("Software", "Citrix Workspace"),
    ("Software", "SAP GUI"),
    ("Software", "AutoCAD"),
    ("Software", "PowerShell"),
    ("Software", "VS Code"),

    # --- Web / Backend Services ---
    ("Web", "Intranet"),
    ("Web", "Reverse Proxy"),
    ("Web", "WebAPI"),
    ("Web", "SSRS Reports"),
    ("Web", "SharePoint"),
    ("Web", "CMS"),

    # --- Database ---
    ("Database", "MSSQL"),
    ("Database", "Oracle"),
    ("Database", "PostgreSQL"),
    ("Database", "MySQL"),
    ("Database", "Redis"),
    ("Database", "ElasticSearch"),

    # --- Storage / Backup ---
    ("Storage", "Fileserver"),
    ("Storage", "NAS"),
    ("Backup", "Veeam"),
    ("Backup", "Snapshot Restore"),
    ("Backup", "Tape Library"),

    # --- Security ---
    ("Security", "Endpoint AV"),
    ("Security", "EDR"),
    ("Security", "SIEM"),
    ("Security", "Email Security"),
    ("Security", "DLP"),
    ("Security", "Certificate Services"),

    # --- Cloud ---
    ("Cloud", "Azure Functions"),
    ("Cloud", "Azure Storage"),
    ("Cloud", "AWS S3"),
    ("Cloud", "AWS Lambda"),

    # --- Messaging ---
    ("Messaging", "Exchange Online"),
    ("Messaging", "SMTP Relay"),
    ("Messaging", "IMAP/POP3"),

    # --- DevOps ---
    ("DevOps", "GitLab CI"),
    ("DevOps", "Jenkins"),
    ("DevOps", "Docker Registry"),
    ("DevOps", "Kubernetes"),
]

impacts  = ["1-High","2-Medium","3-Low"]
urgencies= ["1-High","2-Medium","3-Low"]
# Status-Verteilung: 90% Gelöst, 10% verteilt auf andere
statuses = ["Gelöst","Offen","Abgebrochen","Zurückgewiesen"]
status_weights = [0.9, 0.05, 0.03, 0.02]

priority_map = {1:"Critical",2:"High",3:"Moderate",4:"Low",5:"Planning"}

def priority_from_matrix(impact: str, urgency: str):
    i = safe_parse_level(impact, default=3)
    u = safe_parse_level(urgency, default=3)

    matrix = {
        (1,1):1,(1,2):2,(1,3):3,
        (2,1):2,(2,2):3,(2,3):4,
        (3,1):3,(3,2):4,(3,3):5
    }

    lvl = matrix[(i, u)]
    return lvl, priority_map[lvl]


def derive_issue_type(category: str, service: str, error_code: str) -> str:
    ec = (error_code or "").upper()
    if "00000709" in ec:
        return "printer_spooler_error"
    if "TLS" in ec and "HANDSHAKE" in ec:
        return "vpn_tls_handshake_failed"
    if "SQLSTATE" in ec:
        return "db_sqlstate_error"
    if "ORA-" in ec:
        return "oracle_db_error"
    if "MFA" in ec:
        return "mfa_error"
    if "AD" in category.upper():
        return "ad_access_issue"
    base = f"{category}_{service}".lower().replace(" ", "_")
    return base

# ------------------ OLLAMA CALLS ------------------

def call_ollama_generate(model: str, prompt: str, temperature: float = 0.8, phase: str = "", key: str = "") -> str:

    url = f"{OLLAMA_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
        },
    }
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()
        duration = time.time() - start

        raw_text = text
        if text.startswith("```"):
            text = text.strip("` \n")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()

        # ➜ neue Metriken extrahieren und loggen
        metrics = extract_ollama_metrics(
            data=data,
            model=model,
            phase=phase,
            key=key,
            wall_time_s=duration,
        )
        log_ollama_metrics(metrics)

        # dein bisheriges Logging nicht verlieren:
        log_call(phase, key, duration, len(raw_text), ok=True, error="")
        logging.info(
            f"{phase.upper()} {key}: Dauer {duration:.2f}s, Länge {len(raw_text)} Zeichen, "
            f"Tokens/s={metrics.tokens_per_s:.2f}"
        )
        return text
    except Exception as e:
        duration = time.time() - start
        err_msg = str(e)
        log_call(phase, key, duration, 0, ok=False, error=err_msg)
        logging.error(f"{phase.upper()} {key}: Fehler nach {duration:.2f}s: {err_msg}")
        raise

def safe_json_loads(text: str):
    """
    Robust JSON-Parsing mit minimaler Bereinigung.
    """
    first_brace = text.find("{")
    last_brace  = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        text = text[first_brace:last_brace+1]
    return json.loads(text)

# ------------------ TICKET GENERATION (LLAMA) ------------------

def generate_ticket(category: str, service: str, os_name: str, idx: int):
    system = (
        "Du bist ein erfahrener IT-Helpdesk-Agent in einem Enterprise-Umfeld. "
        "Du generierst realistische IT-Störungstickets (Incidents) auf Deutsch. "
        "Gib ausschließlich ein einzelnes JSON-Objekt im gewünschten Format zurück."
    )

    user = f"""
    Erzeuge ein neues IT-Incident als JSON mit folgenden Feldern:

    - title (kurzer Titel, deutsch)
    - description (1–3 Absätze, aus Sicht des Benutzers; darf laienhaft, emotional oder technisch sein)
    - category ("{category}")
    - service ("{service}")
    - os ("{os_name}")
    - error_code (optional, aber plausibel)
    - impact (1-High, 2-Medium, 3-Low)
    - urgency (1-High, 2-Medium, 3-Low)
    - status (Gelöst, Offen, Abgebrochen, Zurückgewiesen)

    - conversation:
        Liste von 3–10 Nachrichten in zeitlicher Reihenfolge.
        Jede Nachricht hat:
        - author: "kunde" oder "support"
        - content: kurzer deutscher Text

    WICHTIGE RANDPARAMETER:

    1. Variation im technischen Wissen der Anwender:
    - Laienhafte, chaotische oder fehlerhafte Beschreibungen sind erlaubt
    - Sehr technische Nutzer (Power-User) sind ebenfalls erlaubt
    - Rechtschreibfehler, Umgangssprache oder Abkürzungen sind erlaubt

    2. Beschreibung darf fragmentiert oder missverständlich sein:
    - Benutzer kann falsche Diagnosen äußern
    - Benutzer kann Emotionen zeigen („geht plötzlich nix mehr“)

    3. Ticketverlauf soll realistisch sein:
    - Support fragt nach Details
    - Benutzer liefert neue Infos oder widersprüchliche Angaben
    - Support gibt Hinweise / Workarounds / Diagnoseschritte
    - Verlauf kann 3–10 Schritte haben

    4. Kontext darf Enterprise-lastig sein:
    - Optional Hostname (z. B. ws-1234.corp.local)
    - Optional Erwähnung von Standort, Homeoffice, Meeting-Druck etc.

    5. KEINE Erklärtexte, KEINE Prosa außerhalb des JSON-Outputs.

    Gib ausschließlich ein JSON-Objekt zurück.
    """
    prompt = system + "\n\n" + user
    text = call_ollama_generate(
        TICKET_MODEL,
        prompt,
        temperature=0.9,
        phase="ticket",
        key=f"ticket_{idx}"
    )
    data = safe_json_loads(text)
    return data

def ticket_to_row(ticket_dict, ticket_id_int: int):
    created_at = random_created_at()
    impact = ticket_dict.get("impact") or random.choice(impacts)
    urgency = ticket_dict.get("urgency") or random.choice(urgencies)

    lvl, prio = priority_from_matrix(impact, urgency)
    status = ticket_dict.get("status") or random.choices(statuses, weights=status_weights)[0]

    conversation = ticket_dict.get("conversation", [])
    if not isinstance(conversation, list):
        conversation = []

    conv_text = "\n".join(
        f"{m.get('author','?')}: {m.get('content','')}" for m in conversation
    )

    category = ticket_dict.get("category", "")
    service  = ticket_dict.get("service", "")
    os_name  = ticket_dict.get("os", "")

    error_code = ticket_dict.get("error_code", "") or ""
    issue_type = derive_issue_type(category, service, error_code)

    row = {
        "ticket_id": sn_inc_id(ticket_id_int),
        "title": ticket_dict.get("title", ""),
        "description": ticket_dict.get("description", ""),
        "created_at": fmt_iso(created_at),
        "impact": impact,
        "urgency": urgency,
        "priority_level": lvl,
        "priority": prio,
        "status": status,
        "category": category,
        "service": service,
        "category_path": make_category_path(category, service) if category and service else "",
        "ci_id": sn_ci_id(ticket_id_int % 500),
        "os": os_name,
        "hostname": rand_hostname(),
        "reporter": rand_name(),
        "assigned_group": random.choice(GROUPS),
        "assignee": rand_name(),
        "site": random.choice(SITES),
        "conversation_history": json.dumps(conversation, ensure_ascii=False),
        "comments_count": len(conversation),
        "error_code": error_code,
        "gold_kb_id": "",          # später befüllbar, wenn du KB -> Tickets mappst
        "gold_resolution": "",
        "issue_type": issue_type,
        "ticket_fulltext": (
            f"Titel: {ticket_dict.get('title','')}\n"
            f"Beschreibung: {ticket_dict.get('description','')}\n\n"
            f"Verlauf:\n{conv_text}\n"
            f"Status: {status}, Priorität: {prio}, Impact: {impact}, Urgency: {urgency}"
        ),
    }
    return row

def generate_many_tickets(num_tickets: int, out_csv: str):
    fieldnames = [
        "ticket_id","title","description","created_at",
        "impact","urgency","priority_level","priority",
        "status","category","service","category_path",
        "ci_id","os","hostname","reporter",
        "assigned_group","assignee","site",
        "conversation_history","comments_count",
        "error_code","gold_kb_id","gold_resolution",
        "issue_type","ticket_fulltext"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i in range(1, num_tickets+1):
            category, service = random.choice(CATEGORIES_SERVICES)
            os_name = random.choice(OSES)

            try:
                ticket = generate_ticket(category, service, os_name, i, num_tickets)
                row = ticket_to_row(ticket, i)
                w.writerow(row)
            except Exception as e:
                logging.error(f"Ticket {i}: Übersprungen wegen Fehler: {e}")

            if i % 20 == 0:
                logging.info(f"[Tickets] {i}/{num_tickets} generiert")

    logging.info(f"Tickets gespeichert in: {out_csv}")

# ------------------ KB GENERATION (QWEN) ------------------

def build_kb_for_group(group_key, tickets_sample):
    """
    group_key: (category, service, err_key)
    tickets_sample: Liste von dicts (Zeilen aus CSV)
    """
    category, service, err_key = group_key

    ticket_summaries = []
    os_values = []
    error_values = set()

    for t in tickets_sample:
        desc = (t["description"] or "").strip().replace("\n", " ")
        if len(desc) > 200:
            desc = desc[:200] + "..."
        ticket_summaries.append(
            f"- ID: {t['ticket_id']}, Titel: {t['title']}\n"
            f"  Beschreibung: {desc}\n"
            f"  Fehlercode: {t.get('error_code','')}"
        )
        if t.get("os"):
            os_values.append(str(t["os"]))
        if t.get("error_code"):
            error_values.add(str(t["error_code"]))

    tickets_block = "\n".join(ticket_summaries)

    system = (
        "Du bist ein erfahrener IT-Systemingenieur und verfasst Knowledge-Base-Artikel "
        "für einen internen IT-Service-Desk. Du erhältst mehrere ähnliche Tickets "
        "und sollst daraus einen allgemeinen KB-Artikel ableiten. "
        "Antworte ausschließlich als JSON gemäß Schema."
    )

    user = f"""
Hier sind mehrere Incident-Tickets einer ähnlichen Störung:

{tickets_block}

Erzeuge daraus einen allgemeinen Knowledge-Base-Artikel, der dieses Problem beschreibt,
nicht die einzelnen Fälle. Verwende deutschsprachige, aber technisch präzise Formulierungen.

JSON-Schema für die Antwort:

{{
  "title": "Kurzbeschreibung der Störung",
  "category": "z. B. Network, Hardware, Software, Access, Database, Backup, Security",
  "service": "z. B. VPN, Printer, Office, AD, MSSQL, Intranet, etc.",
  "problem": "Allgemeine Problembeschreibung",
  "symptoms": ["Symptom 1", "Symptom 2", "..."],
  "root_cause": ["Mögliche Ursache 1", "Mögliche Ursache 2", "..."],
  "troubleshooting_steps": ["Schritt 1", "Schritt 2", "..."],
  "resolution_steps": ["Abschließender Lösungs-/Fix-Schritt"],
  "references": [
    "z. B. Herstellerhandbuch zum Produkt",
    "z. B. interner SOP-Link (generisch formuliert)"
  ],
  "kb_fulltext": "Fließtext (2–5 Absätze), der Problem, Symptome, Ursachen und Lösung beschreibt."
}}

Randbedingungen:
- Nutze category = "{category}" und service = "{service}" als Basis.
- Der Artikel soll allgemein gültig sein, nicht benutzerspezifisch.
- Gib ausschließlich ein JSON-Objekt im oben beschriebenen Format zurück.
"""

    prompt = system + "\n\n" + user
    text = call_ollama_generate(
        KB_MODEL,
        prompt,
        temperature=0.7,
        phase="kb",
        key=f"{category}_{service}_{err_key or 'none'}"
    )
    data = safe_json_loads(text)

    # Meta aus Tickets ableiten
    os_value = "mixed"
    if os_values:
        # Einfach häufigsten OS-Wert verwenden
        os_value = max(set(os_values), key=os_values.count)
    related_error_codes = ",".join(sorted(error_values)) if error_values else ""

    # Tags aus category/service
    tags = [category.lower(), service.lower()]

    return data, os_value, ",".join(tags), related_error_codes

def generate_kb_from_tickets(incidents_csv: str, kb_out_csv: str):
    df = pd.read_csv(incidents_csv)

    def norm_err(e):
        e = str(e) if not pd.isna(e) else ""
        return e.strip()[:32]  # einfach normalisiert

    df["err_key"] = df["error_code"].apply(norm_err)

    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (row.get("category",""), row.get("service",""), row.get("err_key",""))
        groups[key].append(row.to_dict())

    logging.info(f"Anzahl Gruppen für KB: {len(groups)}")

    kb_rows = []
    kb_counter = 1000000

    for key, tickets in groups.items():
        category, service, err_key = key
        if not category or not service:
            continue

        sample = tickets[:MAX_TICKETS_PER_KB_CONTEXT]

        try:
            kb_data, os_value, tags, rel_err = build_kb_for_group(key, sample)
        except Exception as e:
            logging.error(f"[KB] Fehler bei Gruppe {key}: {e}")
            continue

        kb_counter += 1
        kb_id = sn_kb_id(kb_counter)

        kb_rows.append({
            "kb_id": kb_id,
            "title": kb_data.get("title",""),
            "category": kb_data.get("category", category),
            "service": kb_data.get("service", service),
            "problem": kb_data.get("problem",""),
            "symptoms": " | ".join(kb_data.get("symptoms",[])),
            "root_cause": " | ".join(kb_data.get("root_cause",[])),
            "troubleshooting_steps": " | ".join(kb_data.get("troubleshooting_steps",[])),
            "resolution_steps": " | ".join(kb_data.get("resolution_steps",[])),
            "references": " | ".join(kb_data.get("references",[])),
            "kb_fulltext": kb_data.get("kb_fulltext",""),
            "os": os_value,
            "tags": tags,
            "related_error_codes": rel_err
        })

        if len(kb_rows) % 10 == 0:
            logging.info(f"[KB] {len(kb_rows)} KB-Artikel generiert")

    fieldnames = [
        "kb_id","title","category","service","problem",
        "symptoms","root_cause","troubleshooting_steps",
        "resolution_steps","references","kb_fulltext",
        "os","tags","related_error_codes"
    ]
    with open(kb_out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in kb_rows:
            w.writerow(row)

    logging.info(f"KB-Artikel gespeichert in: {kb_out_csv}")

# ------------------ MAIN ------------------

if __name__ == "__main__":
    overall_start = time.time()
    init_log_csv()

    incidents_csv = os.path.join(OUT_DIR, INC_CSV_FILE)
    kb_csv = os.path.join(OUT_DIR, KB_CSV_FILE)

    logging.info("Starte Ticket-Generierung...")
    generate_many_tickets(NUM_TICKETS, incidents_csv)

    logging.info("Starte KB-Generierung aus Tickets...")
    generate_kb_from_tickets(incidents_csv, kb_csv)

    total_duration = time.time() - overall_start
    logging.info(f"GESAMTZEIT: {total_duration:.2f} Sekunden (~{total_duration/60:.2f} Minuten)")
    print(f"Fertig. Gesamtzeit: {total_duration:.2f} Sekunden (~{total_duration/60:.2f} Minuten)")
