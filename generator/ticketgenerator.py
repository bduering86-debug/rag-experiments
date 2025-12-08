#!/usr/bin/env python3
"""
Generate synthetic IT Incidents (Tickets) with a local Ollama model
and write them directly to a CSV file.

- Tickets: e.g. llama3.1:8b-instruct-q4_K_M über Ollama

CSV-Schema für Incidents:

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
"""

import os
import csv
import uuid
import math
import json
import datetime as dt
import requests
import random
import time

from typing import List, Dict, Any
from bin import config as config
from bin.logging_utils import get_logger
from bin import metrics_utils

# ---------------------------------------------------------------------------
# Initialisierung
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Konfiguration (über ENV steuerbar, damit .env / dotenv-ui greift)
# ---------------------------------------------------------------------------

OLLAMA_HOST = config.OllamaConfig().url
OLLAMA_MODEL_INCIDENTS = config.GeneratorConfig().generator_model_incidents

# Gesamtanzahl zu generierender Tickets
TOTAL_TICKETS = config.DataConfig().total_tickets

# Wie viele Tickets pro Ollama-Call generiert werden sollen
TICKETS_PER_CALL = config.DataConfig().tickets_per_call

# Pfad der Ausgabedatei
OUTPUT_CSV_PATH = config.GeneratorConfig().output_dir

# Ausgabedatei
OUTPUT_CSV_FILENAME = OUTPUT_CSV_PATH.rstrip("/") + "/synthetic_incidents_llm_"+OLLAMA_MODEL_INCIDENTS+".csv"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "ticket_generator"+OLLAMA_MODEL_INCIDENTS+".log")

# ---------------------------------------------------------------------------
# statische Daten für Ticket-Generierung, um Zeit bei Generierung zu verringern
# ---------------------------------------------------------------------------

OSES = ["Windows 10","Windows 11", "Windows Server 2019", "Windows Server 2016",]

# Impact & Urgency Optionen -> werden aktuell nicht genutzt
impacts  = ["1-High","2-Medium","3-Low"]
urgencies= ["1-High","2-Medium","3-Low"]

# Status-Verteilung: 90% Gelöst, 10% verteilt auf andere
statuses = ["Gelöst","Offen","Abgebrochen","Zurückgewiesen"]
status_weights = [0.9, 0.05, 0.03, 0.02]
priority_map = {1:"Critical",2:"High",3:"Moderate",4:"Low",5:"Planning"}

userdata = [
    {'hostname': 'COMP-482910', 'reporter': 'Paul Klein',     'id': 19374, 'site': 'Berlin'},
    {'hostname': 'COMP-019384', 'reporter': 'Anna Schmidt',   'id': 58290, 'site': 'München'},
    {'hostname': 'COMP-593027', 'reporter': 'Noah Wolf',      'id': 76015, 'site': 'Hamburg'},
    {'hostname': 'COMP-740192', 'reporter': 'Mara Weber',     'id': 20488, 'site': 'Köln'},
    {'hostname': 'COMP-128503', 'reporter': 'Ben Fischer',    'id': 91832, 'site': 'Remote'},
    {'hostname': 'COMP-904812', 'reporter': 'Chris Becker',   'id': 47590, 'site': 'Stuttgart'},
    {'hostname': 'COMP-335729', 'reporter': 'Dana Hoffmann',  'id': 31746, 'site': 'Dortmund'},
    {'hostname': 'COMP-672104', 'reporter': 'Fatma Wagner',   'id': 82015, 'site': 'Leipzig'},
    {'hostname': 'COMP-208315', 'reporter': 'Olga Meyer',     'id': 60473, 'site': 'Düsseldorf'},
    {'hostname': 'COMP-519807', 'reporter': 'Erik Klein',     'id': 15789, 'site': 'Hannover'},
    {'hostname': 'COMP-847120', 'reporter': 'Lukas Schmidt',  'id': 49057, 'site': 'Berlin'},
    {'hostname': 'COMP-193475', 'reporter': 'Rita Wolf',      'id': 87231, 'site': 'Bremen'},
    {'hostname': 'COMP-660284', 'reporter': 'Paul Becker',    'id': 30951, 'site': 'Köln'},
    {'hostname': 'COMP-401928', 'reporter': 'Mara Klein',     'id': 74209, 'site': 'Hamburg'},
    {'hostname': 'COMP-275619', 'reporter': 'Anna Weber',     'id': 51084, 'site': 'Nürnberg'},
    {'hostname': 'COMP-983012', 'reporter': 'Ben Meyer',      'id': 68423, 'site': 'Remote'},
    {'hostname': 'COMP-507438', 'reporter': 'Chris Wolf',     'id': 23019, 'site': 'Bochum'},
    {'hostname': 'COMP-749205', 'reporter': 'Dana Fischer',   'id': 95172, 'site': 'Frankfurt'},
    {'hostname': 'COMP-316804', 'reporter': 'Noah Wagner',    'id': 47561, 'site': 'Berlin'},
    {'hostname': 'COMP-128947', 'reporter': 'Fatma Schmidt',  'id': 60127, 'site': 'Hannover'},
    {'hostname': 'COMP-560129', 'reporter': 'Olga Hoffmann',  'id': 36254, 'site': 'München'},
    {'hostname': 'COMP-892014', 'reporter': 'Erik Weber',     'id': 78341, 'site': 'Stuttgart'},
    {'hostname': 'COMP-230598', 'reporter': 'Paul Meyer',     'id': 17892, 'site': 'Köln'},
    {'hostname': 'COMP-741203', 'reporter': 'Rita Becker',    'id': 91507, 'site': 'Berlin'},
    {'hostname': 'COMP-198407', 'reporter': 'Lukas Wagner',   'id': 28413, 'site': 'Leipzig'},
    {'hostname': 'COMP-673015', 'reporter': 'Anna Klein',     'id': 70954, 'site': 'Düsseldorf'},
    {'hostname': 'COMP-259781', 'reporter': 'Chris Schmidt',  'id': 96170, 'site': 'Hamburg'},
    {'hostname': 'COMP-904317', 'reporter': 'Dana Meyer',     'id': 55320, 'site': 'Remote'},
    {'hostname': 'COMP-341892', 'reporter': 'Ben Wolf',       'id': 43621, 'site': 'Frankfurt'},
    {'hostname': 'COMP-782054', 'reporter': 'Noah Hoffmann',  'id': 82091, 'site': 'Berlin'},
    {'hostname': 'COMP-156902', 'reporter': 'Fatma Klein',    'id': 60278, 'site': 'Nürnberg'},
    {'hostname': 'COMP-620487', 'reporter': 'Olga Wolf',      'id': 14705, 'site': 'Bochum'},
    {'hostname': 'COMP-874201', 'reporter': 'Erik Fischer',   'id': 51984, 'site': 'Stuttgart'},
    {'hostname': 'COMP-239510', 'reporter': 'Rita Schmidt',   'id': 70156, 'site': 'München'},
    {'hostname': 'COMP-708312', 'reporter': 'Paul Weber',     'id': 93058, 'site': 'Remote'},
    {'hostname': 'COMP-492038', 'reporter': 'Mara Wolf',      'id': 84120, 'site': 'Berlin'},
    {'hostname': 'COMP-915720', 'reporter': 'Lukas Meyer',    'id': 23751, 'site': 'Hamburg'},
    {'hostname': 'COMP-347820', 'reporter': 'Ben Becker',     'id': 56078, 'site': 'Frankfurt'},
    {'hostname': 'COMP-120698', 'reporter': 'Anna Hoffmann',  'id': 80439, 'site': 'Bremen'},
    {'hostname': 'COMP-584013', 'reporter': 'Chris Weber',    'id': 15728, 'site': 'Berlin'},
    {'hostname': 'COMP-762901', 'reporter': 'Dana Klein',     'id': 63209, 'site': 'München'},
    {'hostname': 'COMP-408215', 'reporter': 'Noah Schmidt',   'id': 27041, 'site': 'Hamburg'},
    {'hostname': 'COMP-936502', 'reporter': 'Fatma Becker',   'id': 88014, 'site': 'Köln'},
    {'hostname': 'COMP-251708', 'reporter': 'Olga Wagner',    'id': 96570, 'site': 'Stuttgart'},
    {'hostname': 'COMP-690134', 'reporter': 'Erik Hoffmann',  'id': 13489, 'site': 'Dortmund'},
    {'hostname': 'COMP-873420', 'reporter': 'Rita Meyer',     'id': 50827, 'site': 'Remote'},
    {'hostname': 'COMP-314907', 'reporter': 'Lukas Wolf',     'id': 74168, 'site': 'Berlin'},
    {'hostname': 'COMP-559802', 'reporter': 'Mara Becker',    'id': 89310, 'site': 'Hamburg'},
    {'hostname': 'COMP-742196', 'reporter': 'Paul Schmidt',   'id': 42056, 'site': 'Frankfurt'},
    {'hostname': 'COMP-805249', 'reporter': 'Anna Becker',    'id': 67012, 'site': 'Leipzig'}
]

it_assignees = [
    {'group': 'IT Service Desk',         'assignee': 'Martin Köhler'},
    {'group': 'Network Operations',      'assignee': 'Tobias Neumann'},
    {'group': 'Security Team',           'assignee': 'Svenja Brandt'},
    {'group': 'ClientSupport',           'assignee': 'Kevin Schulz'},
    {'group': 'Application Support',     'assignee': 'Laura Bergmann'},
    {'group': 'Database Administration', 'assignee': 'Nikolai Richter'},
    {'group': 'Cloud Services Team',     'assignee': 'Melanie Schröder'},
    {'group': 'DevOps Team',             'assignee': 'Jonas Falk'},
    {'group': 'Helpdesk Level',          'assignee': 'Patrick Hoff'},
    {'group': 'Infrastructure Team',     'assignee': 'Daniel Krause'},
    {'group': 'Email Support Team',      'assignee': 'Sarah Krüger'},
    {'group': 'Mobile Device Support',   'assignee': 'Julia Pfeiffer'}
]

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

category_service_assignees = [
    # --- Network ---
    {'category': 'Network', 'service': 'VPN', 'assignee': 'Tobias Neumann'},
    {'category': 'Network', 'service': 'DNS', 'assignee': 'Tobias Neumann'},
    {'category': 'Network', 'service': 'DHCP', 'assignee': 'Tobias Neumann'},
    {'category': 'Network', 'service': 'Firewall', 'assignee': 'Tobias Neumann'},
    {'category': 'Network', 'service': 'Proxy', 'assignee': 'Tobias Neumann'},
    {'category': 'Network', 'service': 'Load Balancer', 'assignee': 'Tobias Neumann'},
    {'category': 'Network', 'service': 'WLAN','assignee': 'Tobias Neumann'},
    {'category': 'Network', 'service': 'Switching', 'assignee': 'Tobias Neumann'},
    {'category': 'Network', 'service': 'Routing', 'assignee': 'Tobias Neumann'},
    {'category': 'Network', 'service': 'TLS/SSL', 'assignee': 'Tobias Neumann'},

    # --- Access & Identity ---
    {'category': 'Access', 'service': 'AD Login', 'assignee': 'Kevin Schulz'},
    {'category': 'Access', 'service': 'LDAP', 'assignee': 'Kevin Schulz'},
    {'category': 'Access', 'service': 'SSO', 'assignee': 'Kevin Schulz'},
    {'category': 'Access', 'service': 'MFA', 'assignee': 'Kevin Schulz'},
    {'category': 'Access', 'service': 'Password Reset', 'assignee': 'Martin Köhler'}, 
    {'category': 'Access', 'service': 'Kerberos', 'assignee': 'Kevin Schulz'},
    {'category': 'Access', 'service': 'Azure AD', 'assignee': 'Kevin Schulz'},
    {'category': 'Access', 'service': 'Conditional Access', 'assignee': 'Kevin Schulz'},

    # --- Hardware / Workplace ---
    {'category': 'Hardware', 'service': 'Printer', 'assignee': 'Patrick Hoff'},
    {'category': 'Hardware', 'service': 'Scanner', 'assignee': 'Patrick Hoff'},
    {'category': 'Hardware', 'service': 'ThinClient', 'assignee': 'Patrick Hoff'},
    {'category': 'Hardware', 'service': 'Monitor', 'assignee': 'Patrick Hoff'},
    {'category': 'Hardware', 'service': 'Dockingstation', 'assignee': 'Patrick Hoff'},
    {'category': 'Hardware', 'service': 'Webcam', 'assignee': 'Patrick Hoff'},
    {'category': 'Hardware', 'service': 'Keyboard','assignee': 'Patrick Hoff'},
    {'category': 'Hardware', 'service': 'Headset', 'assignee': 'Patrick Hoff'},

    # --- Software / Desktop / Client ---
    {'category': 'Software', 'service': 'Office', 'assignee': 'Laura Bergmann'},
    {'category': 'Software', 'service': 'Outlook','assignee': 'Laura Bergmann'},
    {'category': 'Software', 'service': 'Teams', 'assignee': 'Laura Bergmann'},
    {'category': 'Software', 'service': 'Browser','assignee': 'Laura Bergmann'},
    {'category': 'Software', 'service': 'PDF Viewer', 'assignee': 'Laura Bergmann'},
    {'category': 'Software', 'service': 'Antivirus Client', 'assignee': 'Svenja Brandt'},
    {'category': 'Software', 'service': 'VPN Client', 'assignee': 'Tobias Neumann'}, 
    {'category': 'Software', 'service': 'Java Runtime','assignee': 'Laura Bergmann'},
    {'category': 'Software', 'service': 'Citrix Workspace', 'assignee': 'Laura Bergmann'},
    {'category': 'Software', 'service': 'SAP GUI','assignee': 'Laura Bergmann'},
    {'category': 'Software', 'service': 'AutoCAD','assignee': 'Laura Bergmann'},
    {'category': 'Software', 'service': 'PowerShell', 'assignee': 'Laura Bergmann'},
    {'category': 'Software', 'service': 'VS Code','assignee': 'Laura Bergmann'},

    # --- Web / Backend Services ---
    {'category': 'Web', 'service': 'Intranet','assignee': 'Laura Bergmann'},
    {'category': 'Web', 'service': 'Reverse Proxy','assignee': 'Daniel Krause'},
    {'category': 'Web', 'service': 'WebAPI', 'assignee': 'Daniel Krause'},
    {'category': 'Web', 'service': 'SSRS Reports', 'assignee': 'Daniel Krause'},
    {'category': 'Web', 'service': 'SharePoint', 'assignee': 'Laura Bergmann'},
    {'category': 'Web', 'service': 'CMS', 'assignee': 'Laura Bergmann'},

    # --- Database ---
    {'category': 'Database', 'service': 'MSSQL', 'assignee': 'Nikolai Richter'},
    {'category': 'Database', 'service': 'Oracle', 'assignee': 'Nikolai Richter'},
    {'category': 'Database', 'service': 'PostgreSQL', 'assignee': 'Nikolai Richter'},
    {'category': 'Database', 'service': 'MySQL', 'assignee': 'Nikolai Richter'},
    {'category': 'Database', 'service': 'Redis', 'assignee': 'Nikolai Richter'},
    {'category': 'Database', 'service': 'ElasticSearch','assignee': 'Nikolai Richter'},

    # --- Security ---
    {'category': 'Security', 'service': 'Endpoint AV', 'assignee': 'Svenja Brandt'},
    {'category': 'Security', 'service': 'EDR', 'assignee': 'Svenja Brandt'},
    {'category': 'Security', 'service': 'SIEM', 'assignee': 'Svenja Brandt'},
    {'category': 'Security', 'service': 'Email Security', 'assignee': 'Svenja Brandt'},
    {'category': 'Security', 'service': 'DLP', 'assignee': 'Svenja Brandt'},
    {'category': 'Security', 'service': 'Certificate Services', 'assignee': 'Svenja Brandt'},

    # --- Cloud ---
    {'category': 'Cloud', 'service': 'Azure Functions', 'assignee': 'Melanie Schröder'},
    {'category': 'Cloud', 'service': 'Azure Storage', 'assignee': 'Melanie Schröder'},
    {'category': 'Cloud', 'service': 'AWS S3','assignee': 'Melanie Schröder'},
    {'category': 'Cloud', 'service': 'AWS Lambda', 'assignee': 'Melanie Schröder'},

    # --- Messaging ---
    {'category': 'Messaging', 'service': 'Exchange Online', 'assignee': 'Sarah Karüger'},
    {'category': 'Messaging', 'service': 'SMTP Relay', 'assignee': 'Sarah Krüger'},
    {'category': 'Messaging', 'service': 'IMAP/POP3', 'assignee': 'Sarah Krüger'},

    # --- DevOps ---
    {'category': 'DevOps', 'service': 'GitLab CI', 'assignee': 'Jonas Falk'},
    {'category': 'DevOps', 'service': 'Jenkins', 'assignee': 'Jonas Falk'},
    {'category': 'DevOps', 'service': 'Docker Registry','assignee': 'Jonas Falk'},
    {'category': 'DevOps', 'service': 'Kubernetes', 'assignee': 'Jonas Falk'}
]

# Mapping laden -> Kategorien/Srvicea/Assignee in Dictionary umwandeln
ASSIGNEE_MAPPING = {
    (c.get('category', ''), c.get('service', '')): c.get('assignee', 'IT Service Desk')
    for c in category_service_assignees
}

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logger = get_logger("ticketgenerator_"+OLLAMA_MODEL_INCIDENTS)


# ---------------------------------------------------------------------------
# TicketGenerator
# ---------------------------------------------------------------------------

class TicketGenerator:
    """
    Generiert synthetische Tickets in Batches über Ollama
    und schreibt sie in eine CSV-Datei.
    """

    # ------------------------------------------------------------------
    # CSV STruktur festlegen
    # ------------------------------------------------------------------
    CSV_FIELDS = [
        "ticket_id",
        "title",
        "description",
        "created_at",
        "impact",
        "urgency",
        "priority_level",
        "priority",
        "status",
        "category",
        "service",
        "category_path",
        "ci_id",
        "os",
        "hostname",
        "reporter",
        "assigned_group",
        "assignee",
        "site",
        "conversation_history",
        "comments_count",
        "error_code",
        "gold_kb_id",
        "gold_resolution",
        "issue_type",
        "ticket_fulltext",
    ]

    def __init__(
        self,
        base_url: str,
        model: str,
        total_tickets: int,
        tickets_per_call: int,
        output_csv_path: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.total_tickets = total_tickets
        self.tickets_per_call = tickets_per_call
        self.output_csv_path = OUTPUT_CSV_FILENAME

        if self.tickets_per_call <= 0:
            raise ValueError("tickets_per_call muss > 0 sein")

        logger.info(
            "Initialisiert TicketGenerator: total_tickets=%s, tickets_per_call=%s, model=%s",
            self.total_tickets,
            self.tickets_per_call,
            self.model,
        )

    
    # ------------------------------------------------------------------
    # Hilfsfunktionen
    # ------------------------------------------------------------------
    
    # Assignee basierend auf Kategorie/Service ermitteln
    def get_assignee(self, category: str, service: str) -> str:
        assignee = "n/a"  # Default
        for c in category_service_assignees:
            if c["category"] == category and c["service"] == service:
                assignee = c["assignee"]
        # return inkl. Fallback
        return assignee or "Jon Doe"

    # assigned_group basierend auf Assignee ermitteln
    def get_group_for_assignee(self, assignee: str) -> str:
        for entry in it_assignees:
            if entry["assignee"] == assignee:
                return entry["group"]
        return "IT Service Desk"   # Fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Generiert alle Tickets und schreibt sie direkt in die CSV-Datei.
        """
        num_batches = math.ceil(self.total_tickets / self.tickets_per_call)
        logger.info(
            "Starte Ticket-Generierung: %s Tickets in %s Batches à max. %s Tickets.",
            self.total_tickets,
            num_batches,
            self.tickets_per_call,
        )

        # CSV initialisieren
        file_exists = os.path.exists(self.output_csv_path)
        with open(self.output_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)

            if not file_exists:
                writer.writeheader()
                logger.debug("CSV-Header geschrieben nach %s", self.output_csv_path)

            remaining = self.total_tickets
            batch_index = 0
            generated_so_far = 0

            logger.debug("CSV "+ OUTPUT_CSV_FILENAME +" in %s", self.output_csv_path)
            logger.debug("Batch size vor while %s", self.tickets_per_call)
            while remaining > 0:
                batch_index += 1
                current_batch_size = min(self.tickets_per_call, remaining)

                logger.info(
                    "[Batch %s/%s] Generiere %s Tickets (remaining: %s)...",
                    batch_index,
                    num_batches,
                    current_batch_size,
                    remaining,
                )

                try:
                    tickets = self._generate_ticket_batch(current_batch_size)
                except Exception as e:
                    logger.exception("Fehler bei Batch %s: %s", batch_index, e)
                    break

                if len(tickets) != current_batch_size:
                    logger.warning(
                        "Batch %s: Erwartet %s Tickets, erhalten %s.",
                        batch_index,
                        current_batch_size,
                        len(tickets),
                    )

                # JSON -> CSV-Zeilen schreiben
                for t in tickets:
                    row = self._ticket_to_csv_row(t)
                    writer.writerow(row)
                    generated_so_far += 1

                logger.info(
                    "[Batch %s/%s] Batch abgeschlossen. Generierte Tickets gesamt: %s",
                    batch_index,
                    num_batches,
                    generated_so_far,
                )

                remaining -= current_batch_size

        logger.info(
            "Ticket-Generierung abgeschlossen. Insgesamt generierte Tickets: %s",
            generated_so_far,
        )

    # ------------------------------------------------------------------
    # Intern: Ein Batch über Ollama
    # ------------------------------------------------------------------
    def _generate_ticket_batch(
        self,
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """
        Ruft Ollama einmal auf und lässt sich batch_size Tickets generieren.
        Erwartet, dass das Modell ein JSON-Array von Ticket-Objekten zurückgibt.
        """

        # OS, Kategorie/Service und assignee für Prompt auswählen
        os_prompt                       = random.choice(OSES)
        category_prompt, service_prompt = random.choice(CATEGORIES_SERVICES)
        #impacts_prompt                  = random.choice(impacts)
        #urgencies_prompt                = random.choice(urgencies)
        #statuses_prompt                 = random.choices(statuses, weights=status_weights, k=1)[0]
        assignee_prompt                 = self.get_assignee(category_prompt, service_prompt)
        assignee_group_prompt           = self.get_group_for_assignee(assignee_prompt)
        reporter_list                   = random.sample(userdata, batch_size)

        prompt_arguments = {
            "os_prompt": os_prompt,
            "category_prompt": category_prompt,
            "service_prompt": service_prompt,
            #"impacts_prompt": impacts_prompt,
            #"urgencies_prompt": urgencies_prompt,
            #"statuses_prompt": statuses_prompt,
            "assignee_prompt": assignee_prompt,
            "assigned_group_prompt": assignee_group_prompt,
            "prompt_reporter": reporter_list
        }

        prompt = self._build_prompt_for_batch(batch_size, **prompt_arguments)

        # Prompt vollständig loggen (DEBUG-Level, damit Logs nicht explodieren)
        logger.debug("Prompt für Batch (size=%s):\n%s", batch_size, prompt)

        # Ollama aufruf
        response_text, eval_tokens, prompt_tokens, duration = self._call_ollama(prompt)

        # Response (gekürzt) mitloggen
        logger.debug(
            "Response für Batch (size=%s), first 1000 chars:\n%s",
            batch_size,
            response_text[:1000],
        )

        # Metriken an die zentrale Metrics-Utility melden
        metrics_utils.log_ollama_call(
            batch_size=batch_size,
            duration=duration,
            eval_tokens=eval_tokens,
            prompt_tokens=prompt_tokens,
        )

        logger.info(
        "Ollama-Call abgeschlossen: batch_size=%s, duration=%.2fs, tokens/s=%.2f",
        batch_size,
        duration,
        (eval_tokens / duration) if duration and eval_tokens else 0.0,
        )
        

        tickets = self._parse_batch_response(response_text)
        logger.debug("Parsed Tickets im Batch: %s", len(tickets))
        return tickets

    def _build_prompt_for_batch(self, batch_size: int, **prompt_arguments) -> str:
        """
        Baut einen kompakten Prompt für GENAU batch_size Tickets mit definierten Grenzen,
        damit die Antwort kurz, präzise und token-effizient bleibt.
        """

        # Singular / Plural vorbereiten
        if batch_size == 1:
            ticket_count_phrase = "EXAKT 1 deutschsprachiges IT-Incident-Ticket"
            array_phrase = "als JSON-Array mit GENAU EINEM Objekt (genau einem Ticket)"
            ticket_word = "Ticket"
            ticket_word_additional = "Das"
            varianz_text = (
                "Sonstiges:\n"
                "- Nutzerwissen variieren (Laie, durchschnittlich, Power-User)."
            )
        else:
            ticket_count_phrase = f"EXAKT {batch_size} deutschsprachige IT-Incident-Tickets"
            array_phrase = "als JSON-Array"
            ticket_word = "Tickets"
            ticket_word_additional = "Jedes"
            varianz_text = (
                f"Varianzanforderungen:\n"
                f"- Alle {batch_size} Tickets müssen sich klar unterscheiden.\n"
                f"- Variiere Situation, Ursache, Symptome, Tonfall, Nutzerwissen und Formulierungen.\n"
                f"- Keine identischen Sätze zwischen verschiedenen Tickets.\n"
                f"ZUSÄTZLICHE VARIANZANFORDERUNGEN (WICHTIG):\n"
                f"- Alle Titel müssen sich in Wortwahl UND Struktur unterscheiden.\n"
                f"- Die Beschreibungen müssen unterschiedliche Situationen darstellen "
                f"(z. B. Fehlercodes, Symptome, Nutzeraktionen).\n"
                f"- Die gold_resolution MUSS inhaltlich je Ticket verschieden sein "
                f"(andere Ursache, andere Lösungsschritte).\n"
                f"- issue_type MUSS gesetzt werden (z. B. \"AuthenticationError\", "
                f"\"ConnectionFailure\", \"Timeout\", \"ClientBug\", \"ServerMisconfiguration\").\n"
                f"- error_code MUSS entweder realistisch wirken (\"0x80070005\", "
                f"\"ERR_PROXY_CONNECTION_FAILED\") oder \"\" sein – aber NICHT in allen Tickets gleich.\n"
                f"- Vermeide generische Formulierungen wie \"Ein Update für das Tool ist erforderlich\".\n"
                f"- Die Texte müssen deutlich voneinander abweichen; erkennbare Wiederholungen sind NICHT erlaubt."
            )


        reporter_compact = [
            {"r": u["reporter"], "h": u["hostname"], "s": u["site"]}
            for u in prompt_arguments["prompt_reporter"]
        ]

        prompt = f"""
        Du erzeugst realistische IT-Incident-Tickets für ein ITSM-System.

        Erzeuge {ticket_count_phrase} {array_phrase}. Keine Erklärungen, kein Text außerhalb des JSON-Arrays.
        Die Antwort MUSS mit "[" beginnen und mit "]" enden.

        Feste Vorgaben für JEDES {ticket_word}:
        - Verwende grammatikalisch möglichst korrektes Hochdeutsch.
        - category: "{prompt_arguments['category_prompt']}"
        - service: "{prompt_arguments['service_prompt']}"
        - os: "{prompt_arguments['os_prompt']}"
        - impact: eine Auswahl aus ["1-High","2-Medium","3-Low"] -> nur die Zahl
        - urgency: eine Auswahl aus ["1-High","2-Medium","3-Low"] -> nur die Zahl
        - priority_level / priority: gemäß Matrix {priority_map} -> nur die Zahl
        - status: immer "Gelöst"
        - assignee: "{prompt_arguments['assignee_prompt']}"
        - assigned_group: "{prompt_arguments['assigned_group_prompt']}"
        Verwende für jedes Ticket GENAU EIN Element aus der folgenden Liste und WEISE DIE FELDER korrekt zu:
        - reporter = Wert aus "r"
        - hostname = Wert aus "h"
        - site = Wert aus "s"
        Die Auswahl für jedes Ticket MUSS aus dieser Liste stammen, und jedes Element darf mehrfach verwendet werden:
        {reporter_compact}

        Inhalte / Stil:
        - title:
        - 3–6 Wörter
        - für jedes Ticket unterschiedlich und klar unterscheidbar
        - description:
        - 1–2 kurze Sätze, realistisch, verschiedene Formulierungen
        - max. ca. 20 Wörter pro Satz
        - conversation_history:
            - Immer "" (leer lassen) -> erzeuge keine Gesprächshistorie
        - ticket_fulltext: immer "" (leer lassen)
        - gold_kb_id: immer "" (leer lassen)
        - comments_count:
            - Immer "" (leer lassen)
        - gold_resolution: GENAU 1 kurzer Satz (max. 20 Wörter) zur Lösung. Muss konkrete Maßnahmen nennen, z. B. Konfigurationsänderungen, Registry-Anpassungen, Dienstneustarts, Patch-Nummern, Berechtigungen.
         Muss einer dieser Kategorien entsprechen:
        ["AuthenticationError", "ConnectivityIssue", "PermissionDenied", "Timeout", "ClientBug", "Misconfiguration", "OutOfMemory", "ServiceUnavailable"]
        error_code:
        - Kann einer dieser Varianten entsprechen:
        ["0x80070005", "ERR_SSL_VERSION", "ERR_PROXY_CONNECTION_FAILED", "0x80004005", "404", "503", ""]

        {varianz_text}

        Ausgabeformat:
        Gib AUSSCHLIESSLICH ein JSON-Array zurück, ohne ```-Codeblock, ohne Erklärungstext.
        Verwende EXAKT diese Schlüssel für ein Ticket-Objekt:

        "title",
        "description",
        "impact",
        "urgency",
        "priority_level",
        "priority",
        "status",
        "category",
        "service",
        "category_path",
        "ci_id",
        "os",
        "hostname",
        "reporter",
        "assigned_group",
        "assignee",
        "site",
        "conversation_history",
        "comments_count",
        "error_code",
        "gold_kb_id",
        "gold_resolution",
        "issue_type",
        "ticket_fulltext"
            """.strip()

        return prompt


    def _call_ollama(self, prompt: str) -> tuple[str, int, int, float]:
        """
        Ruft das lokale Ollama-API (/api/chat) mit dem gegebenen Prompt auf.
        Erwartet eine nicht-streamende Antwort.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Du bist ein Assistent, der strukturierte Incident-Tickets im JSON-Format erzeugt."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "options": {
                "temperature": config.GeneratorConfig.generator_temperature,
                "top_p": config.GeneratorConfig.generator_top_p, 
                "num_ctx": config.GeneratorConfig.generator_ctx_tokens,
                "repeat_penalty": config.GeneratorConfig.generator_repeat_penalty,
                "num_predict": config.GeneratorConfig.generator_num_predict,
                "seed": config.GeneratorConfig.generator_seed,
            }
        }

        logger.debug("Sende Request an Ollama: %s", url)
        
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=600)
        duration = time.time() - t0

        if not resp.ok:
            logger.error(
                "Ollama-Request fehlgeschlagen: Status=%s, Text=%s",
                resp.status_code,
                resp.text[:500],
            )
            resp.raise_for_status()

        data = resp.json()
        # /api/chat-Response: {"message": {"role": "...", "content": "..."}, ...}
        content = data.get("message", {}).get("content", "")

        # Token-Metriken auslesen (Ollama: eval_count / prompt_eval_count)
        eval_tokens_raw = data.get("eval_count")
        prompt_tokens_raw = data.get("prompt_eval_count")

        try:
            eval_tokens = int(eval_tokens_raw) if eval_tokens_raw is not None else 0
        
        except (TypeError, ValueError):
            eval_tokens = 0

        try:
            prompt_tokens = int(prompt_tokens_raw) if prompt_tokens_raw is not None else 0
        
        except (TypeError, ValueError):
            prompt_tokens = 0

        if not content:
            logger.warning("Leere Antwort oder kein 'content' im Ollama-Response.")

        return content, eval_tokens, prompt_tokens, duration


    def strip_json_codeblock(self, text: str) -> str:
        """
        Entfernt ```json / ``` code fences und liefert reines JSON zurück.
        """
        text = text.strip()

        # Case 1: Antwort beginnt mit ```json oder ``` 
        if text.startswith("```"):
            # Entferne erstes ```
            text = text.lstrip("`")

            # Entferne führendes "json" falls vorhanden
            if text.lower().startswith("json"):
                text = text[4:].lstrip()

            # Entferne abschließendes ``` falls vorhanden
            end = text.rfind("```")
            if end != -1:
                text = text[:end]

        # Zusätzlich ersten JSON-Block herauslösen
        start_candidates = [text.find("["), text.find("{")]
        start_candidates = [i for i in start_candidates if i != -1]
        if not start_candidates:
            raise ValueError("Kein JSON-Start gefunden")

        start = min(start_candidates)

        end_candidates = [text.rfind("]"), text.rfind("}")]
        end_candidates = [i for i in end_candidates if i != -1]
        if not end_candidates:
            raise ValueError("Kein JSON-Ende gefunden")

        end = max(end_candidates) + 1

        return text[start:end].strip()

    def _parse_batch_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parsed die Modellantwort (JSON-Array) in Python-Objekte.
        """

        logger.debug("Roh-Response (gekürzt): %s", response_text[:1000])

        try:
            clean = self.strip_json_codeblock(response_text)
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error("JSON-Parsing fehlgeschlagen: %s", e)
            logger.debug("Roh-Response (gekürzt): %s", response_text[:2000])
            return []

        if not isinstance(data, list):
            logger.error("Erwartet wurde ein JSON-Array, erhalten: %s", type(data))
            return []

        tickets: List[Dict[str, Any]] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                logger.warning("Eintrag %s ist kein Objekt, wird übersprungen.", idx)
                continue
            tickets.append(item)

        return tickets

    # ------------------------------------------------------------------
    # JSON → CSV-Row
    # ------------------------------------------------------------------
    def _ticket_to_csv_row(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wandelt ein Ticket-JSON-Objekt in eine CSV-Zeile mit allen erwarteten Feldern.
        Fehlende Keys werden sinnvoll gefüllt.
        """
        # Zeitzonenbewusste UTC-Zeit statt deprecated utcnow()
        now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Ticket-ID generieren, wenn nicht vorhanden
        ticket_id = ticket.get("ticket_id") or str(uuid.uuid4())

        # created_at ggf. aus Ticket, sonst jetzt
        created_at = ticket.get("created_at") or now_iso

        # Helper für get mit Default
        def g(key: str, default: str = "") -> Any:
            v = ticket.get(key)
            if v is None:
                return default
            return v

        row: Dict[str, Any] = {
            "ticket_id": ticket_id,
            "title": g("title"),
            "description": g("description"),
            "created_at": created_at,
            "impact": g("impact"),
            "urgency": g("urgency"),
            "priority_level": g("priority_level"),
            "priority": g("priority"),
            "status": g("status", "New"),
            "category": g("category"),
            "service": g("service"),
            "category_path": g("category_path"),
            "ci_id": g("ci_id"),
            "os": g("os"),
            "hostname": g("hostname"),
            "reporter": g("reporter"),
            "assigned_group": g("assigned_group"),
            "assignee": g("assignee"),
            "site": g("site"),
            "conversation_history": g("conversation_history"),
            "comments_count": g("comments_count", 0),
            "error_code": g("error_code"),
            "gold_kb_id": g("gold_kb_id", ""),
            "gold_resolution": g("gold_resolution", ""),
            "issue_type": g("issue_type"),
            "ticket_fulltext": g("ticket_fulltext"),
        }

        return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Starte Ticketgenerator-Skript.")
    logger.info(
        "Konfiguration: OLLAMA_HOST=%s, MODEL=%s, TOTAL_TICKETS=%s, TICKETS_PER_CALL=%s, OUTPUT_CSV=%s",
        OLLAMA_HOST,
        OLLAMA_MODEL_INCIDENTS,
        TOTAL_TICKETS,
        TICKETS_PER_CALL,
        OUTPUT_CSV_FILENAME,
    )

    # Metrics-Run starten
    run_id = metrics_utils.start_run(
        model=OLLAMA_MODEL_INCIDENTS,
        total_tickets=TOTAL_TICKETS,
        tickets_per_call=TICKETS_PER_CALL,
        temperature=config.GeneratorConfig.generator_temperature,
        top_p=config.GeneratorConfig.generator_top_p, 
        ctx_tokens=config.GeneratorConfig.generator_ctx_tokens,
        repeat_penalty=config.GeneratorConfig.generator_repeat_penalty,
        seed=config.GeneratorConfig.generator_seed,
        num_predict=config.GeneratorConfig.generator_num_predict
    )
    

    generator = TicketGenerator(
        base_url=OLLAMA_HOST,
        model=OLLAMA_MODEL_INCIDENTS,
        total_tickets=TOTAL_TICKETS,
        tickets_per_call=TICKETS_PER_CALL,
        output_csv_path=OUTPUT_CSV_PATH,
    )

    generator.run()

    # Metrics-Run beenden (Summary-Log)
    metrics_utils.end_run()

    logger.info("Skript beendet. run_id=%s", run_id)

if __name__ == "__main__":
    main()
