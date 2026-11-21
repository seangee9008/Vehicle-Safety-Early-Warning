"""Read the parts of an agent brief the AI agent tab shows. The brief text itself is never changed;
these only find sections, counts and short labels in it."""

from __future__ import annotations

import re

SECTION_TITLES = {"failure pattern": "What drivers report", "severity": "Severity", "evidence": "Evidence",
                  "watchlist reasons check": "Flag reasons check",
                  "similar past recalls": "Similar past recalls", "suggested next step": "Suggested next step"}

# Common defect symptoms -> a short label (under 8 words). The brief's failure pattern gives its
# chips: specific symptoms in the order they appear, then the generic ones.
SYMPTOMS = [
    (r"phantom|false activation|without (any )?(physical )?obstruction|sudden deceleration",
     "Brakes slam on with nothing ahead"),
    (r"emergency braking|\bAEB\b", "Emergency braking misfires"),
    (r"unintended acceleration|uncommanded acceleration|unexpected vehicle movement", "Car moves or speeds up unexpectedly"),
    (r"stall", "Engine stalls while driving"),
    (r"loss of (motive |engine )?power|propulsion loss|losses of motive power|shut ?down|shutdowns",
     "Loses power while driving"),
    (r"transmission|clutch|gear hunting|rough shifting|downshift|forward gears", "Transmission fails or shifts badly"),
    (r"blackout|go(es)? (dark|blank)|black screen", "Screens or gauges go dark"),
    (r"infotainment|touchscreen", "Infotainment screen fails"),
    (r"battery (depletion|drain)|parasitic", "Battery drains while parked"),
    (r"unable to start|fail(s)? to restart|difficult(ies)? restarting|prevent vehicle restart", "Won't restart"),
    (r"warning (lights|messages)|error codes|trouble codes|check engine", "Warning lights and error codes"),
    (r"glare|reflection", "Dashboard glare blocks the view"),
    (r"steering", "Steering gets stiff or sticks"),
    (r"camera", "Cameras stop working"),
    (r"lane keep|lane departure|cruise control|driver.assist|blind.spot", "Driver-assist features fail"),
    (r"software update", "Starts after software updates"),
    (r"turn signal", "Turn signals misbehave"),
    (r"misfire|rough idl", "Misfires and rough idle"),
    (r"limp mode|reduced acceleration", "Drops into reduced power mode"),
    (r"engine (failure|breakdown)|lifters|camshaft|rod bearing", "Engine breaks down"),
    (r"overheat", "Parts overheat"),
    (r"rollover", "Rollover reported"),
    (r"smoke", "Smoke reported"),
    (r"low mileage|first few weeks|shortly after delivery|under [\d,]+ miles", "Fails soon after purchase"),
    (r"electrical", "Electrical failures"),
    (r"glass breakage|sunroof", "Glass breaks"),
]


GENERIC = {"Electrical failures", "Starts after software updates", "Fails soon after purchase"}


def sections(markdown: str) -> dict[str, str]:
    out, title, body = {}, None, []
    for line in str(markdown).splitlines():
        if line.startswith("## "):
            if title is not None:
                out[title] = "\n".join(body).strip()
            key = re.sub(r"\s*\(.*\)$", "", line[3:].strip()).lower()
            title, body = SECTION_TITLES.get(key, line[3:].strip()), []
        else:
            body.append(line)
    if title is not None:
        out[title] = "\n".join(body).strip()
    return out


def severity(text: str) -> dict[str, int]:
    counts = {}
    for label in ("Crashes", "Fires", "Injuries", "Deaths"):
        m = re.search(rf"{label}\s*:\s*(\d+)", text or "", flags=re.IGNORECASE)
        counts[label.lower()] = int(m.group(1)) if m else 0
    return counts


def bullets(text: str) -> list[str]:
    return [line.lstrip("-* ").strip() for line in (text or "").splitlines() if line.strip().startswith(("-", "*"))]


def citation_ids(text: str) -> list[str]:
    return [n for group in re.findall(r"\[ODINO[^\]]*\]", text) for n in re.findall(r"\d+", group)]


def strip_citations(text: str) -> str:
    text = re.sub(r"\s*\[ODINO[^\]]*\](,\s*)?", " ", text)
    return re.sub(r"\s+([.,;])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()


def reason_checks(text: str) -> list[tuple[str, str, str]]:
    """(reason, verdict SUPPORT|CONTEXT|CONTRADICT, why) for each line of the reasons check."""
    out = []
    for line in bullets(text):
        m = re.match(r"(.+?):\s*(SUPPORT|CONTEXT|CONTRADICT)\b\s*(.*)$", line, flags=re.IGNORECASE)
        if m:
            out.append((m.group(1).strip(), m.group(2).upper(), m.group(3).strip(" ()")))
    return out


def similar_recalls(text: str) -> tuple[int, bool]:
    """(number of past recalls listed, whether the agent found none relevant)."""
    listed = len(bullets(text))
    return listed, listed == 0


def symptom_chips(failure_pattern: str, n: int = 3) -> list[str]:
    found = {}
    for pattern, label in SYMPTOMS:
        m = re.search(pattern, failure_pattern or "", flags=re.IGNORECASE)
        if m and label not in found:
            found[label] = (label in GENERIC, m.start())
    chips = sorted(found, key=found.get)[:n]  # specific symptoms first, in the order they appear
    if len(chips) < n:  # fall back to the opening words of each sentence
        for sentence in re.split(r"(?<=[.!?])\s+", failure_pattern or ""):
            words = re.sub(r"^(Drivers|Owners|Complainants)( of [^,]*?)? (report|experience|describe)s? ",
                           "", sentence).split()
            if len(words) >= 3 and len(chips) < n:
                text = " ".join(words[:7]).rstrip(",.;") + ("…" if len(words) > 7 else "")
                chips.append(text[:1].upper() + text[1:])
    return chips[:n]
