#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KBGenerator
-----------
Grundlegende Struktur wurde mit Hilfe von ChatGPT erstellt auf entsprechende Anweisung im Austausch in Form von Vibe-Coding.
Einzelne Code-Abschnitte wurden manuell angepasst und erweitert oder um Imports ergänzt.
Erzeugt aus synthetischen Incident-Tickets konsistente Wissensartikel (KB-Artikel).

Pipeline:
1. Tickets aus CSV laden
2. Tickets nach kb_key gruppieren (z.B. category|service|issue_type|error_code)
3. Pro Gruppe eine repräsentative Untermenge auswählen (z.B. max. 10 Tickets)
4. Pro Gruppe GENAU EINEN KB-Artikel vom LLM erzeugen lassen (JSON)
5. KB-Artikel in kb_csv schreiben
6. Tickets um gold_kb_id ergänzen und in tickets_with_kb.csv schreiben
"""

import csv
import json
import time
import uuid
import random
import requests

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
from bin import config as config

# Logging- und Metrics-Utility importieren (manuell ergänzt)
from bin.logging_utils import get_logger
from bin import metrics_utils

logger = get_logger("kb_generator")

#----------------------------
# Hilfsfunktionen
#----------------------------

def _normalize_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [", ".join(str(v) for v in value.values())]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, dict):
                out.append(", ".join(str(v) for v in item.values()))
            else:
                out.append(str(item))
        return out
    return [str(value)]

def _normalize_to_str_list(value) -> list[str]:
    """
    Nimmt beliebige Werte (str, list, dict, None, etc.)
    und gibt eine Liste sinnvoller Strings zurück.
    Leere Objekte werden entfernt.
    """
    if value is None:
        return []

    # Einfacher String
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []

    # Dict -> "key=value, key2=value2" (falls leer: ignorieren)
    if isinstance(value, dict):
        if not value:  # leeres {}
            return []
        joined = ", ".join(f"{k}={v}" for k, v in value.items())
        joined = joined.strip()
        return [joined] if joined else []

    # Liste -> rekursiv über alle Elemente
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_normalize_to_str_list(item))
        # Alles, was leer ist, rauswerfen
        return [s for s in out if s.strip()]

    # Fallback: alles andere als String
    s = str(value).strip()
    return [s] if s else []


#----------------------------
# Hauptklassen
#----------------------------
@dataclass
class KBGeneratorConfig:
    tickets_csv: Path
    output_kb_csv: Path
    output_tickets_with_kb_csv: Path

    ollama_host: str
    model: str

    max_tickets_per_prompt: int = getattr(config.GeneratorConfig(), 'generator_tickets_for_kb_context', 10)

    # LLM-Parameter -> nachträglich angepasst, damit die Konfiguration aus der config.py genutzt werden kann
    temperature: float = config.GeneratorConfig().generator_temperature
    top_p: float = config.GeneratorConfig().generator_top_p
    repeat_penalty: float = config.GeneratorConfig().generator_repeat_penalty
    ctx_tokens: int = config.GeneratorConfig().generator_kb_ctx_tokens
    num_predict: config.GeneratorConfig().generator_num_predict = None 


@dataclass
class KBArticle:
    kb_id: str
    title: str
    category: str
    service: str
    issue_type: str
    error_codes: List[str]
    environment: str
    problem: str
    symptoms: List[str]
    root_cause: str
    resolution_steps: List[str]
    validation: str
    related_ticket_ids: List[str] = field(default_factory=list)
    kb_fulltext: str = "" 

    @classmethod
    @classmethod
    def from_llm_json(cls, data: Dict[str, Any]) -> "KBArticle":
        error_codes = _normalize_to_str_list(data.get("error_codes", []))
        symptoms = _normalize_to_str_list(data.get("symptoms", []))
        root_cause_list = _normalize_to_str_list(data.get("root_cause", []))
        resolution_steps = _normalize_to_str_list(data.get("resolution_steps", []))
        related_ids = _normalize_to_str_list(data.get("related_ticket_ids", []))

        article = cls(
            kb_id=data.get("kb_id", ""),
            title=data.get("title", ""),
            category=data.get("category", ""),
            service=data.get("service", ""),
            issue_type=data.get("issue_type", ""),
            error_codes=error_codes,
            environment=data.get("environment", {}) or {},
            problem=data.get("problem", ""),
            symptoms=symptoms,
            root_cause=" | ".join(root_cause_list),   # intern als String oder:
            # root_cause=root_cause_list,            # falls du es als Liste speichern willst
            resolution_steps=resolution_steps,
            validation=data.get("validation", ""),
            related_ticket_ids=related_ids,
            kb_fulltext=data.get("kb_fulltext", ""),
        )

        if not article.kb_fulltext:
            article.kb_fulltext = cls.build_fulltext(article)

        return article



    def to_csv_row(self) -> Dict[str, str]:
        return {
            "kb_id": self.kb_id,
            "title": self.title,
            "category": self.category,
            "service": self.service,
            "issue_type": self.issue_type,
            "error_codes": " | ".join(_normalize_to_str_list(self.error_codes)),
            "environment": (
                self.environment
                if isinstance(self.environment, str)
                else json.dumps(self.environment, ensure_ascii=False)
            ),
            "problem": self.problem,
            "symptoms": " | ".join(_normalize_to_str_list(self.symptoms)),
            "root_cause": " | ".join(_normalize_to_str_list(self.root_cause)),
            "resolution_steps": " | ".join(_normalize_to_str_list(self.resolution_steps)),
            "validation": self.validation,
            "related_ticket_ids": " | ".join(_normalize_to_str_list(self.related_ticket_ids)),
            "kb_fulltext": self.kb_fulltext,
        }


    @staticmethod
    def build_fulltext(article: "KBArticle") -> str:
        parts: List[str] = []

        if article.kb_id:
            parts.append(f"KB-ID: {article.kb_id}")
        if article.title:
            parts.append(f"Titel: {article.title}")
        if article.category or article.service:
            parts.append(f"Kategorie/Service: {article.category} / {article.service}")
        if article.issue_type:
            parts.append(f"Issue-Typ: {article.issue_type}")
        if article.error_codes:
            parts.append("Fehlercodes: " + ", ".join(_normalize_to_str_list(article.error_codes)))
        if article.problem:
            parts.append("Problem: " + article.problem)
        if article.symptoms:
            parts.append("Symptome: " + " | ".join(_normalize_to_str_list(article.symptoms)))
        if article.root_cause:
            parts.append("Ursache: " + " | ".join(_normalize_to_str_list(article.root_cause)))
        if article.resolution_steps:
            parts.append("Lösungsschritte: " + " | ".join(_normalize_to_str_list(article.resolution_steps)))
        if article.validation:
            parts.append("Validierung: " + article.validation)



        # nur nicht-leere Teile zusammenführen
        return "\n".join(p for p in parts if p.strip())

class KBGenerator:
    def __init__(self, config: KBGeneratorConfig) -> None:
        self.cfg = config
        self.ollama_host = config.ollama_host
        self.model = config.model

        # Metriken-Run initialisieren

        metrics_utils.start_run(
            model=self.model,
            total_tickets=0,        # hier: Anzahl Tickets ist optional -> wird nicht genutzt oder kann bei Bedarf gesetzt werden auf Gesamtanzahl der Tickets
            tickets_per_call=0,  # hier: Anzahl Tickets pro KB-Call (repräsentativ) -> wird nicht genutzt oder kann bei Bedarf gesetzt werden auf Tickets pro KB-Call
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            ctx_tokens=self.cfg.ctx_tokens,
            repeat_penalty=self.cfg.repeat_penalty,
            seed=None,
            num_predict=self.cfg.num_predict
        )

        logger.info(
            "Initialisiere KBGenerator: tickets_csv=%s, output_kb_csv=%s, "
            "output_tickets_with_kb_csv=%s, model=%s",
            self.cfg.tickets_csv,
            self.cfg.output_kb_csv,
            self.cfg.output_tickets_with_kb_csv,
            self.model,
        )

    # ----------------------------
    # High-Level Pipeline
    # ----------------------------

    def run(self) -> None:
        start_time = time.time()

        tickets = self._load_tickets()
        logger.info("Geladene Tickets: %s", len(tickets))

        if not tickets:
            logger.warning("Keine Tickets geladen – breche KB-Generierung ab.")
            return

        groups = self._group_tickets_by_kb_key(tickets)
        logger.info("Gruppierte Tickets in %s kb_key-Gruppen.", len(groups))

        kb_articles: List[KBArticle] = []
        ticket_to_kb_id: Dict[str, str] = {}

        # --- KB-CSV vorbereiten (Header einmal schreiben) ---
        output_kb = Path(self.cfg.output_kb_csv)
        output_kb.parent.mkdir(parents=True, exist_ok=True)

        kb_fieldnames = [
            "kb_id",
            "title",
            "category",
            "service",
            "issue_type",
            "error_codes",
            "environment",
            "problem",
            "symptoms",
            "root_cause",
            "resolution_steps",
            "validation",
            "related_ticket_ids",
            "kb_fulltext",
        ]

        # --- Tickets-mit-KB-CSV vorbereiten (Header einmal schreiben) ---
        output_tickets_with_kb = Path(self.cfg.output_tickets_with_kb_csv)
        output_tickets_with_kb.parent.mkdir(parents=True, exist_ok=True)

        ticket_fieldnames = list(tickets[0].keys())
        if "gold_kb_id" not in ticket_fieldnames:
            ticket_fieldnames.append("gold_kb_id")

        with output_kb.open("w", encoding="utf-8", newline="") as kb_file, \
            output_tickets_with_kb.open("w", encoding="utf-8", newline="") as tickets_file:

            kb_writer = csv.DictWriter(kb_file, fieldnames=kb_fieldnames)
            kb_writer.writeheader()

            tickets_writer = csv.DictWriter(tickets_file, fieldnames=ticket_fieldnames)
            tickets_writer.writeheader()

            # --- Hauptschleife über Gruppen ---
            for i, (kb_key, group_tickets) in enumerate(groups.items(), start=1):
                logger.info(
                    "[Gruppe %s/%s] kb_key=%s, Tickets in Gruppe=%s",
                    i, len(groups), kb_key, len(group_tickets)
                )

                repr_tickets = self._select_representative_tickets(group_tickets)
                logger.debug(
                    "Ausgewählte repräsentative Tickets für kb_key=%s: %s IDs",
                    kb_key, [t["id"] for t in repr_tickets]
                )

                kb_id = f"KB-{uuid.uuid4().hex[:8].upper()}"
                prompt = self._build_prompt_for_group(kb_id, kb_key, repr_tickets)

                logger.debug("Prompt für kb_key=%s (gekürzt): %s", kb_key, prompt[:500])

                kb_json = self._call_ollama_for_kb(prompt, repr_tickets)
                if kb_json is None:
                    logger.warning("Keine KB-Antwort für kb_key=%s – Gruppe wird übersprungen.", kb_key)
                    continue

                logger.debug("Erhaltenes KB-JSON für kb_key=%s: %s", kb_key, kb_json)

                kb_article = KBArticle.from_llm_json(kb_json)

                # Fallback: falls das Modell kb_id nicht korrekt setzt, unsere nehmen
                if not kb_article.kb_id:
                    kb_article.kb_id = kb_id

                # Fulltext sicherstellen (falls from_llm_json ihn nicht schon gebaut hat)
                if not getattr(kb_article, "kb_fulltext", ""):
                    kb_article.kb_fulltext = KBArticle.build_fulltext(kb_article)

                kb_articles.append(kb_article)

                # Direkt nach dem Append: Zeile in die CSV schreiben
                kb_row = kb_article.to_csv_row()
                kb_writer.writerow(kb_row)

                logger.debug("KB-Artikel in CSV geschrieben: %s", kb_row["kb_id"])

                # Alle Tickets der Gruppe bekommen diese KB-ID
                for t in group_tickets:
                    ticket_to_kb_id[t["id"]] = kb_article.kb_id

                    # 👉 Ticket sofort mit KB-ID in Tickets-CSV schreiben
                    row = dict(t)
                    row["gold_kb_id"] = kb_article.kb_id
                    tickets_writer.writerow(row)

        duration = time.time() - start_time
        logger.info(
            "KB-Generierung abgeschlossen. KB-Artikel: %s, Tickets: %s, Dauer: %.2fs",
            len(kb_articles), len(tickets), duration
        )

        metrics_utils.end_run()
        logger.info("Skript beendet.")  # run_id kannst du bei Bedarf wieder ergänzen

    # ----------------------------
    # Step 1: Tickets laden
    # ----------------------------

    def _load_tickets(self) -> List[Dict[str, Any]]:
        tickets: List[Dict[str, Any]] = []
        tickets_csv = Path(self.cfg.tickets_csv)
        with tickets_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Stelle sicher, dass eine ID da ist
                if not row.get("id"):
                    # Je nach deinem Schema ggf. "ticket_id" etc.
                    row["id"] = row.get("ticket_id") or str(uuid.uuid4())
                tickets.append(row)
        return tickets

    # ----------------------------
    # Step 2: Gruppierung
    # ----------------------------

    def _build_kb_key(self, ticket: Dict[str, Any]) -> str:
        category = (ticket.get("category") or "").strip()
        service = (ticket.get("service") or "").strip()
        issue_type = (ticket.get("issue_type") or "").strip()
        error_code = (ticket.get("error_code") or "NONE").strip()

        # Du kannst das bei Bedarf erweitern (z. B. um os oder ci_id)
        return f"{category}|{service}|{issue_type}|{error_code}"

    def _group_tickets_by_kb_key(
        self, tickets: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for t in tickets:
            key = self._build_kb_key(t)
            groups.setdefault(key, []).append(t)
        return groups

    # ----------------------------
    # Step 3: repräsentative Tickets
    # ----------------------------

    def _select_representative_tickets(
        self, tickets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Wählt eine repräsentative Untermenge von Tickets für den KB-Prompt aus.

        Heuristik:
        - Bei <= max_tickets_per_prompt: alle
        - Sonst nach Impact/Urgency sortieren
        - Ein paar Tickets aus Anfang/Mitte/Ende + Zufallsstichprobe
        """

        max_n = self.cfg.max_tickets_per_prompt
        if len(tickets) <= max_n:
            return tickets

        def sort_key(t: Dict[str, Any]):
            impact = str(t.get("impact") or "")
            urgency = str(t.get("urgency") or "")
            return (impact, urgency, t.get("id") or "")

        sorted_tickets = sorted(tickets, key=sort_key)

        # Indices vorne / Mitte / hinten
        indices = [
            0,
            len(sorted_tickets) // 3,
            (2 * len(sorted_tickets)) // 3,
            len(sorted_tickets) - 1,
        ]
        picked = set()
        subset: List[Dict[str, Any]] = []

        for idx in indices:
            if 0 <= idx < len(sorted_tickets) and idx not in picked:
                subset.append(sorted_tickets[idx])
                picked.add(idx)

        # Rest zufällig auffüllen
        remaining = [t for i, t in enumerate(sorted_tickets) if i not in picked]
        random.shuffle(remaining)
        subset.extend(remaining[: max_n - len(subset)])

        return subset

    # ----------------------------
    # Step 4: Prompt bauen
    # ----------------------------

    def _build_prompt_for_group(
    self,
    kb_id: str,
    kb_key: str,
    repr_tickets: List[Dict[str, Any]],
    ) -> str:
        """
        Baut den Prompt für die KB-Generierung aus einer Ticket-Gruppe.

        kb_key hat das Format: "<category>|<service>|<issue_type>|<error_code>"
        repr_tickets ist eine Liste repräsentativer Tickets dieser Gruppe.
        """

        # kb_key aufsplitten (robust gegen fehlende Teile)
        parts = kb_key.split("|")
        category = parts[0] if len(parts) > 0 else ""
        service = parts[1] if len(parts) > 1 else ""
        issue_type = parts[2] if len(parts) > 2 else ""
        error_code = parts[3] if len(parts) > 3 and parts[3] != "NONE" else ""

        # Tickets für den Prompt kompakt aufbereiten
        tickets_for_prompt: List[Dict[str, Any]] = []
        related_ids: List[str] = []

        for t in repr_tickets:
            tid = t.get("id") or t.get("ticket_id")
            if tid:
                related_ids.append(str(tid))

            tickets_for_prompt.append(
                {
                    "id": tid,
                    "title": t.get("title", ""),
                    "description": t.get("description", ""),
                    "impact": t.get("impact", ""),
                    "urgency": t.get("urgency", ""),
                    "priority": t.get("priority", ""),
                    "status": t.get("status", ""),
                    "os": t.get("os", ""),
                    "site": t.get("site", ""),
                    "error_code": t.get("error_code", ""),
                    "gold_resolution": t.get("gold_resolution", ""),
                }
            )

        tickets_json = json.dumps(tickets_for_prompt, ensure_ascii=False, indent=2)
        related_ids_json = json.dumps(list(dict.fromkeys(related_ids)), ensure_ascii=False)

        prompt = f"""
    Du bist ein erfahrener ITSM-Wissensdatenbank-Autor.

    Du erhältst mehrere Incident-Tickets, die zum gleichen technischen Problem gehören
    (gleiche Kategorie, Service, issue_type und ggf. error_code).

    Ausgangsdaten zur Problemklasse:
    - category: "{category}"
    - service: "{service}"
    - issue_type: "{issue_type}"
    - error_code: "{error_code}"

    Beispiel-Tickets zu diesem Problem (nur Kontext, NICHT zurückgeben):
    {tickets_json}

    Deine Aufgabe:
    1. Analysiere die Tickets und leite EIN übergreifendes technisches Problem ab.
    2. Formuliere einen wiederverwendbaren Wissensartikel für die ITSM-Knowledgebase.
    3. Beschreibe:
    - das allgemeine Problem,
    - typische Symptome,
    - mögliche Ursachen,
    - empfohlene Lösungsschritte,
    - wie die Lösung verifiziert werden kann.

    ERWARTETE FELDER IM JSON-OBJEKT:

    Gib GENAU EIN JSON-Objekt mit den folgenden Schlüsseln zurück:

    - "kb_id": String
        * Verwende bevorzugt die gegebene KB-ID: "{kb_id}"
    - "title": String
        * Prägnanter Titel, der das Problem beschreibt.
    - "category": String
        * Verwende die gegebene Kategorie: "{category}"
    - "service": String
        * Verwende den gegebenen Service: "{service}"
    - "issue_type": String
        * Verwende den gegebenen issue_type: "{issue_type}"
    - "error_codes": Liste von Strings
        * Entweder [ "{error_code}" ] falls sinnvoll
        * oder [] (leere Liste), wenn kein spezifischer Code wichtig ist.
    - "environment": String
        * Kurzbeschreibung der betroffenen Umgebung (z. B. Betriebssysteme, Applikationstyp, typische Kontexte).
    - "problem": String
        * Zusammenfassung des technischen Kernproblems in 2–4 Sätzen.
    - "symptoms": Liste von Strings
        * 3–6 allgemeine Symptome (z. B. Fehlermeldungen, beobachtetes Verhalten).
        * KEINE Ticket-IDs, KEINE Ticket-Titel, KEINE Objekte oder Dictionaries.
        * Jedes Element muss ein einfacher String sein.
    - "root_cause": Liste von Strings
        * 1–4 mögliche Ursachen in Stichpunkten.
        * Jedes Element muss ein einfacher String sein.
    - "resolution_steps": Liste von Strings
        * Konkrete, geordnete Lösungsschritte (3–7 Einträge, jeweils 1–2 Sätze).
        * Jedes Element muss ein einfacher String sein.
    - "validation": String
        * Wie wird überprüft, dass das Problem wirklich gelöst ist (1–3 Sätze).
    - "related_ticket_ids": Liste von Strings
        * NUR die IDs der Tickets, die zu dieser Problemklasse gehören.
        * Verwende GENAU diese IDs: {related_ids_json}
    - "kb_fulltext": dieses Feld SOLL NICHT gesetzt werden.
        * LASS dieses Feld komplett weg. Es wird später im System automatisch erzeugt.

    FORMATREGELN (SEHR WICHTIG):

    - Antworte mit GENAU EINEM JSON-Objekt.
    - KEIN JSON-Array, KEINE zusätzliche Ebene.
    - KEINE Markdown-Formatierung, KEIN ```json-Codeblock.
    - KEIN Fließtext außerhalb des JSON-Objekts.
    - "symptoms", "root_cause" und "resolution_steps" dürfen ausschließlich Listen von Strings enthalten.
    * KEINE Dictionaries/Objekte, KEINE Ticket-Metadaten.
    - "related_ticket_ids" darf ausschließlich Ticket-IDs als Strings enthalten.

    ANTWORT:
    Gib NUR das JSON-Objekt zurück.
    """.strip()

        return prompt


    # ----------------------------
    # Step 5: Ollama-Call
    # ----------------------------

    def _call_ollama_for_kb(
        self,
        prompt: str,
        tickets_in_group: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Ruft das LLM über Ollama auf und parst das JSON-Objekt für einen KB-Artikel.
        """

        url = f"{self.ollama_host}/api/chat"
        options: Dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "repeat_penalty": self.cfg.repeat_penalty,
            "num_ctx": self.cfg.ctx_tokens,
        }

        # einfache Heuristik für num_predict, falls nicht gesetzt:
        if self.cfg.num_predict is not None:
            options["num_predict"] = self.cfg.num_predict
        else:
            # grobe Abschätzung: ein KB-Artikel ~ 800–1500 Tokens
            options["num_predict"] = 1500

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
        }

        logger.debug("Sende KB-Request an Ollama: url=%s, options=%s", url, options)
        t0 = time.time()
        response = requests.post(url, json=payload, timeout=600)

        duration = time.time() - t0
        response.raise_for_status()
        data = response.json()

        # Ollama Chat-Response: message.content enthält den Text
        content = data.get("message", {}).get("content", "")

        # Metrikdaten, falls verfügbar
        eval_tokens = data.get("eval_count", 0)
        prompt_tokens = data.get("prompt_eval_count", 0)

        tokens_per_sec = eval_tokens / duration if duration > 0 else 0.0
        logger.info(
            "KB-Ollama-Call: duration=%.2fs, eval_tokens=%s, prompt_tokens=%s, tokens/s=%.2f",
            duration, eval_tokens, prompt_tokens, tokens_per_sec
        )

        # Metriken an dein zentrales System melden
        metrics_utils.log_ollama_call(
            #model=self.model,
            eval_tokens=eval_tokens,
            prompt_tokens=prompt_tokens,
            duration=duration,
            batch_size=len(tickets_in_group),
        )

        # JSON aus content parsen
        kb_json = self._parse_kb_json(content)
        return kb_json

    def _parse_kb_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Versucht, den JSON-Text des KB-Artikels robust zu parsen.
        Entfernt ggf. ```json ... ```-Wrapper.
        """

        raw = text.strip()

        # Entferne evtl. ```json ... ```-Blöcke
        if raw.startswith("```"):
            # Naiver Strip für ```json ... ```
            raw = raw.strip("`")
            # Falls noch ein "json" am Anfang steht, entfernen
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        # Versuche direktes json.loads
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("KB-JSON-Parsing fehlgeschlagen: %s", e)
            logger.debug("Roh-Response KB (gekürzt): %s", raw[:1000])
            return None

    # ----------------------------
    # Step 6: Ausgabe schreiben
    # ----------------------------

    def _write_kb_csv(self, kb_articles: List[KBArticle]) -> None:
        output = Path(self.cfg.output_kb_csv)
        output.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "kb_id",
            "title",
            "category",
            "service",
            "issue_type",
            "error_codes",
            "environment",
            "problem",
            "symptoms",
            "root_cause",
            "resolution_steps",
            "validation",
            "related_ticket_ids",
        ]

        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for kb in kb_articles:
                writer.writerow(kb.to_csv_row())

        logger.info("KB-CSV geschrieben: %s (Anzahl KBs: %s)", output, len(kb_articles))

# ----------------------------
# Optionales CLI-Main
# ----------------------------

def main():
    
    cfg = KBGeneratorConfig(

        # Optionen aus der config.py nutzen -> manuell angepasst
        #tickets_csv=Path(config.GeneratorConfig().output_dir+"/synthetic_incidents_llm_phi4-mini:latest.csv"),
        tickets_csv=Path(config.GeneratorConfig().output_dir+"/synthetic_incidents_llm.csv"),
        output_kb_csv=Path(config.GeneratorConfig().output_dir+"/kb_articles_llm.csv"),
        output_tickets_with_kb_csv=Path(config.GeneratorConfig().output_dir+"/synthetic_incidents_with_kb.csv"),
        ollama_host=config.OllamaConfig().url,
        #ollama_host=config.OllamaConfig().url_test,
        model=config.GeneratorConfig().generator_model_knowledgebase,
        max_tickets_per_prompt=config.GeneratorConfig().generator_tickets_for_kb_context,
        temperature=config.GeneratorConfig().generator_kb_temperature,
        top_p=config.GeneratorConfig().generator_kb_top_p,
        repeat_penalty=config.GeneratorConfig().generator_kb_repeat_penalty,
        ctx_tokens=config.GeneratorConfig().generator_kb_ctx_tokens,
        num_predict=config.GeneratorConfig().generator_kb_num_predict
    )

    logger.debug("Starte KBGenerator mit Konfiguration: %s", cfg)

    gen = KBGenerator(cfg)
    gen.run()


if __name__ == "__main__":
    main()
