#!/usr/bin/env python3
import csv, random, json, os, textwrap
from datetime import datetime, timedelta

# ------------------ Konfiguration ------------------
OUT_DIR = "./output"
NUM_INCIDENTS = 50_000
DAYS_BACK = 180
SEED = 888

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ Helper-Funktionen ------------------
def sn_kb_id(n: int) -> str:
    return f"KB{n:07d}"

def sn_inc_id(n: int) -> str:
    return f"INC{n:08d}"

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

OSES = ["Windows 10","Windows 11","Ubuntu 22.04","macOS 12","RHEL 8"]

# ------------------ KB-Templates ------------------
KB_BASE = [
    {
        "category": "Network",
        "service": "VPN",
        "title": "VPN-Verbindung schlägt fehl",
        "problem": "Benutzer können keine VPN-Verbindung herstellen.",
        "symptoms": [
            "Verbindung bricht während des TLS-Handshakes ab",
            "MFA-Bestätigung wird abgelehnt",
            "Client-Log zeigt Zertifikats- oder TLS-Fehler"
        ],
        "root_cause": [
            "Abgelaufenes oder falsches Client-Zertifikat",
            "Falsch verknüpftes MFA-Gerät",
            "Veraltete TLS-Konfiguration am Gateway"
        ],
        "troubleshooting": [
            "Client-Zertifikate in der Zertifikatsverwaltung prüfen",
            "MFA-Gerät im Identity-Portal neu registrieren",
            "VPN-Client und Gateway-Konfiguration auf aktuelle TLS-Version prüfen"
        ],
        "resolution": "Zertifikat erneuert, MFA-Gerät neu verknüpft und TLS-Konfiguration aktualisiert; VPN-Verbindung stabil.",
        "error_pool": ["TLS_HANDSHAKE_FAILED","CERT_EXPIRED"],
        "references": [
            "VPN-Client-Handbuch des Herstellers",
            "TLS-Hardening-Leitfaden der internen IT-Security"
        ]
    },
    {
        "category": "Hardware",
        "service": "Printer",
        "title": "Netzwerkdrucker offline / Fehler 0x00000709",
        "problem": "Benutzer können nicht auf einen freigegebenen Netzwerkdrucker drucken.",
        "symptoms": [
            "Fehler 0x00000709 beim Drucken",
            "Drucker erscheint als offline oder nicht bereit",
            "Druckspooler-Dienst stoppt unerwartet"
        ],
        "root_cause": [
            "Veralteter oder defekter Druckertreiber",
            "Korrupte Druckwarteschlange",
            "Fehlende Abhängigkeiten des Spooler-Dienstes"
        ],
        "troubleshooting": [
            "Druckspooler-Dienst stoppen und Spool-Verzeichnis leeren",
            "Aktuellen Treiber vom Hersteller installieren",
            "Abhängigkeiten des Spooler-Dienstes in der Diensteverwaltung prüfen"
        ],
        "resolution": "Druckertreiber aktualisiert, Warteschlange bereinigt und Spooler-Dienst neu gestartet; Druckfunktion wiederhergestellt.",
        "error_pool": ["0x00000709","SPOOLER_ERROR_1068"],
        "references": [
            "Microsoft-Dokumentation zum Druckspooler",
            "Herstellerhandbuch des Druckers"
        ]
    }
]

# ------------------ KB generieren ------------------
def build_kb():
    kb_entries = []
    kb_id_counter = 1_000_000

    for base in KB_BASE:
        for _ in range(5):  # 5 Varianten pro Basis
            kb_id_counter += 1
            kb = dict(base)
            kb["kb_id"] = sn_kb_id(kb_id_counter)
            kb["os"] = random.choice(OSES)
            kb["tags"] = [kb["category"].lower(),
                           kb["service"].lower(),
                           kb["os"].lower().replace(" ", "")]
            kb["related_error_codes"] = [random.choice(kb["error_pool"])]
            kb["kb_fulltext"] = (
                f"Problem: {kb['problem']}\n"
                f"Symptome: {', '.join(kb['symptoms'])}\n"
                f"Ursache: {', '.join(kb['root_cause'])}\n"
                f"Troubleshooting: {', '.join(kb['troubleshooting'])}\n"
                f"Lösung: {kb['resolution']}\n"
                f"Weiterführende Dokumentation: {', '.join(kb['references'])}"
            )
            kb_entries.append(kb)

    kb_path = os.path.join(OUT_DIR, "synthetic_kb_augmented.csv")
    with open(kb_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "kb_id","title","category","service","os","problem","symptoms",
            "root_cause","troubleshooting_steps","resolution_steps",
            "tags","related_error_codes","kb_fulltext","references"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for kb in kb_entries:
            w.writerow({
                "kb_id": kb["kb_id"],
                "title": kb["title"],
                "category": kb["category"],
                "service": kb["service"],
                "os": kb["os"],
                "problem": kb["problem"],
                "symptoms": " | ".join(kb["symptoms"]),
                "root_cause": " | ".join(kb["root_cause"]),
                "troubleshooting_steps": " | ".join(kb["troubleshooting"]),
                "resolution_steps": kb["resolution"],
                "tags": ",".join(kb["tags"]),
                "related_error_codes": ",".join(kb["related_error_codes"]),
                "kb_fulltext": kb["kb_fulltext"],
                "references": " | ".join(kb["references"])
            })

    return kb_entries, kb_path

# ------------------ Incidents generieren ------------------
impacts  = ["1-High","2-Medium","3-Low"]
urgencies= ["1-High","2-Medium","3-Low"]
statuses = ["Gelöst","Offen","Abgebrochen","Zurückgewiesen"]
priority_map = {1:"Critical",2:"High",3:"Moderate",4:"Low",5:"Planning"}

def priority_from_matrix(impact: str, urgency: str):
    i = int(impact.split("-")[0])
    u = int(urgency.split("-")[0])
    matrix = {
        (1,1):1,(1,2):2,(1,3):3,
        (2,1):2,(2,2):3,(2,3):4,
        (3,1):3,(3,2):4,(3,3):5
    }
    lvl = matrix[(i,u)]
    return lvl, priority_map[lvl]

def derive_issue_type(category: str, service: str, error_code: str) -> str:
    if "00000709" in error_code:
        return "printer_spooler_error"
    if "TLS_HANDSHAKE_FAILED" in error_code:
        return "vpn_tls_handshake_failed"
    return f"{category}_{service}".lower().replace(" ","_")

def generate_conversation(kb: dict, created_at: datetime):
    msgs = []
    t = created_at
    for i in range(random.randint(2,4)):
        t += timedelta(minutes=random.randint(3, 20))
        role = "kunde" if i % 2 == 0 else "support"
        if role == "kunde":
            content = f"Ich habe weiterhin ein Problem mit {kb['service']}."
        else:
            content = "Wir prüfen die Konfiguration und melden uns wieder."
        msgs.append({
            "timestamp": fmt_iso(t),
            "author_role": role,
            "content": content
        })
    return msgs

def build_incidents(kb_entries):
    inc_path = os.path.join(OUT_DIR, "synthetic_incidents_50k_augmented.csv")
    fields = [
        "ticket_id","title","description","created_at","impact","urgency",
        "priority_level","priority","status","category","service","category_path",
        "ci_id","os","hostname","reporter","assigned_group","assignee","site",
        "conversation_history","comments_count","error_code",
        "gold_kb_id","gold_resolution","issue_type","ticket_fulltext"
    ]

    with open(inc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for n in range(1, NUM_INCIDENTS + 1):
            kb = random.choice(kb_entries)
            created_at = random_created_at(DAYS_BACK)
            impact = random.choice(impacts)
            urgency = random.choice(urgencies)
            lvl, prio = priority_from_matrix(impact, urgency)
            status = random.choices(statuses, weights=[0.9,0.05,0.03,0.02])[0]
            has_kb = (status == "Gelöst" and random.random() < 0.9)
            error_code = random.choice(kb["related_error_codes"]) if random.random() < 0.7 else ""

            conv = generate_conversation(kb, created_at)
            conv_text = "\n".join(f"{m['author_role']}: {m['content']}" for m in conv)
            ticket_fulltext = (
                f"Titel: {kb['title']}\n"
                f"Beschreibung: {kb['problem']}\n"
                f"Verlauf:\n{conv_text}\n"
                f"Status: {status}, Priorität: {prio}, Impact: {impact}, Urgency: {urgency}"
            )

            w.writerow({
                "ticket_id": sn_inc_id(n),
                "title": kb["title"],
                "description": kb["problem"],
                "created_at": fmt_iso(created_at),
                "impact": impact,
                "urgency": urgency,
                "priority_level": lvl,
                "priority": prio,
                "status": status,
                "category": kb["category"],
                "service": kb["service"],
                "category_path": make_category_path(kb["category"], kb["service"]),
                "ci_id": sn_ci_id(n % 500),
                "os": kb["os"],
                "hostname": rand_hostname(),
                "reporter": rand_name(),
                "assigned_group": random.choice(GROUPS),
                "assignee": rand_name(),
                "site": random.choice(SITES),
                "conversation_history": json.dumps(conv, ensure_ascii=False),
                "comments_count": len(conv),
                "error_code": error_code,
                "gold_kb_id": kb["kb_id"] if has_kb else "",
                "gold_resolution": kb["resolution"] if has_kb else "",
                "issue_type": derive_issue_type(kb["category"], kb["service"], error_code),
                "ticket_fulltext": ticket_fulltext
            })

    return inc_path

# ------------------ Main ------------------
if __name__ == "__main__":
    kb_entries, kb_path = build_kb()
    inc_path = build_incidents(kb_entries)

    readme_path = os.path.join(OUT_DIR, "README_synthetic_data_augmented.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""
        Synthetic IT KB & Incidents (Augmented, Deutsch)
        ------------------------------------------------
        Verzeichnis: {os.path.abspath(OUT_DIR)}

        Dateien:
        - {kb_path}
        - {inc_path}

        KB:
        - kb_fulltext: kombinierter Text (Problem, Symptome, Ursache, Troubleshooting, Lösung, Doku-Hinweise).

        Incidents:
        - {NUM_INCIDENTS} Tickets mit Ticketverlauf (conversation_history) und ticket_fulltext.
        - issue_type: normalisierte Problemklasse.
        - gold_kb_id/gold_resolution: Verknüpfung zu KB-Artikel für Retrieval-Evaluation.
        """))

    print("Fertig. Daten liegen in:", os.path.abspath(OUT_DIR))
