import re
from typing import List, Any

def safe_parse_level(value, default=3):
    """
    Extrahiert eine Zahl 1–3 aus einem Impact/Urgency-String.
    Beispiele:
    - "1 - Hoch" → 1
    - "2" → 2
    - 3 → 3
    - "3 - Low" → 3s
    - None → default (3)
    - "Hoch" → default (3)
    - "5 - undefined" → capped auf 3
    - "0 - invalid" → default (3)
    """

    if value is None:
        return default

    # immer in string wandeln
    s = str(value).strip()

    # alle Zahlen extrahieren
    numbers = re.findall(r"\d+", s)

    if not numbers:
        return default

    level = int(numbers[0])

    # falls Level außerhalb des 1–3 Bereichs → normalisieren
    if level < 1 or level > 3:
        return default

    return level


def safe_split(value: Any, sep: str = ",") -> List[str]:
    """
    Teilt einen Wert robust in Teile auf:
    - Strings: normal split + strip
    - int/float: in einen String umwandeln
    - None: leere Liste
    """
    if value is None:
        return []

    if isinstance(value, str):
        parts = value.split(sep)
    else:
        # int, float, bool, sonstiges → erst in String umwandeln
        parts = str(value).split(sep)

    return [p.strip() for p in parts if p.strip()]
