"""Browser test of the dashboard (manual test cases TC-01..TC-09, automated).

    streamlit run app.py --server.port 8599          # terminal 1
    python tests/ui_smoke_check.py                     # terminal 2

Needs `pip install playwright`. It uses an installed Chrome/Brave/Edge if found
(set BROWSER_PATH to override), otherwise run `playwright install chromium` once.
Writes outputs/ui_test_report.txt and dashboard screenshots to outputs/screenshots/.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
from edurisk import config as C  # noqa: E402

URL = os.environ.get("EDURISK_URL", "http://localhost:8599/")
SHOTS = C.OUTPUT_DIR / "screenshots"
SAMPLE = C.CD_DIR / "3_Datasets" / "sample_new_students.csv"
CANDIDATES = [
    os.environ.get("BROWSER_PATH", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]
results: list[tuple[str, str, bool, str]] = []


def check(tc: str, title: str, ok: bool, detail: str = "") -> None:
    results.append((tc, title, bool(ok), detail))
    print(f"{tc} {'PASS' if ok else 'FAIL'} - {title} {detail}")


def settle(page, ms=2500):
    page.wait_for_selector("[data-testid='stApp']", timeout=30000)
    page.wait_for_timeout(ms)
    for _ in range(40):
        if page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        page.wait_for_timeout(500)


def interventions() -> int:
    with sqlite3.connect(C.DB_FILE) as con:
        return con.execute("SELECT COUNT(*) FROM intervention").fetchone()[0]


def main() -> int:
    SHOTS.mkdir(parents=True, exist_ok=True)
    exe = next((p for p in CANDIDATES if p and Path(p).exists()), None)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        page.goto(URL)
        settle(page, 4000)
        body = page.inner_text("body")
        check("TC-01", "Dashboard opens with KPI tiles",
              "Students scored" in body and "885" in body and "High risk" in body)
        page.screenshot(path=SHOTS / "dash_1_overview.png")

        page.get_by_text("After Semester 2").click()
        settle(page)
        check("TC-02", "Changing the checkpoint updates the view",
              "Current cohort - After Semester 2" in page.inner_text("body"))
        page.get_by_text("After Semester 1").click()
        settle(page)

        page.get_by_role("tab", name="At-risk list").click()
        settle(page)
        txt = page.inner_text("body")
        check("TC-03", "Risk-tier filter lists High + Medium students", "students match the filters" in txt)
        page.screenshot(path=SHOTS / "dash_2_risk_list.png")
        with page.expect_download() as dl:
            page.get_by_role("button", name="Download list (CSV)").click()
        check("TC-04", "At-risk list downloads as CSV", dl.value.suggested_filename.endswith(".csv"),
              dl.value.suggested_filename)

        page.get_by_role("tab", name="Student profile").click()
        settle(page)
        txt = page.inner_text("body")
        check("TC-05", "Student profile shows probabilities and reasons",
              "Why the model flagged this student" in txt and "Semester records" in txt)
        before = interventions()
        page.get_by_label("Note").fill("UI smoke test - counselling booked")
        page.get_by_role("button", name="Save").click()
        settle(page)
        after = interventions()
        check("TC-06", "Logging an intervention writes to the database", after == before + 1,
              f"({before} -> {after} rows)")
        page.screenshot(path=SHOTS / "dash_3_student_profile.png")

        page.get_by_role("tab", name="Score new students").click()
        settle(page)
        page.locator("input[type='file']").set_input_files(str(SAMPLE))
        settle(page, 5000)
        # Streamlit draws tables on a canvas, so check the widgets instead of cell text
        n_tables = page.locator("[data-testid='stDataFrame']").count()
        has_dl = page.get_by_role("button", name="Download scores (CSV)").count() == 1
        check("TC-07", "Uploading the sample CSV returns risk scores", has_dl and n_tables >= 1,
              f"({n_tables} table)")
        page.screenshot(path=SHOTS / "dash_4_upload_scoring.png")

        bad = C.OUTPUT_DIR / "ui_test_bad_upload.csv"
        bad.write_text("Gender;Age at enrollment\n1;20\n")
        page.locator("input[type='file']").set_input_files(str(bad))
        settle(page, 3000)
        check("TC-08", "Upload with missing columns shows a clear error",
              "Missing columns" in page.inner_text("body"))
        bad.unlink()

        page.get_by_role("tab", name="Model insight").click()
        settle(page)
        txt = page.inner_text("body")
        check("TC-09", "Model insight shows metrics and fairness note",
              "How reliable is the warning?" in txt and "scholarship" in txt)
        page.screenshot(path=SHOTS / "dash_5_model_insight.png", full_page=True)
        browser.close()

    lines = [f"{tc}\t{'PASS' if ok else 'FAIL'}\t{title}\t{detail}" for tc, title, ok, detail in results]
    (C.OUTPUT_DIR / "ui_test_report.txt").write_text("\n".join(lines) + "\n")
    failed = sum(not r[2] for r in results)
    print(f"{len(results) - failed}/{len(results)} UI checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
