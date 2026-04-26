from __future__ import annotations

import glob
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from .utils import ensure_dir, normalize_whitespace, write_json

BASE_URL = "https://www.nseindia.com"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL,
    "Connection": "keep-alive",
}


def read_symbols(symbols_file: str | Path) -> list[str]:
    path_str = str(symbols_file)

    if any(char in path_str for char in ["*", "?"]):
        matched_files = sorted(Path(path) for path in glob.glob(path_str))
        if not matched_files:
            raise FileNotFoundError(f"No symbol files matched pattern: {symbols_file}")

        symbols: list[str] = []
        seen: set[str] = set()
        for matched_file in matched_files:
            with open(matched_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    symbol = line.strip().upper()
                    if not symbol or symbol in seen:
                        continue
                    seen.add(symbol)
                    symbols.append(symbol)
        return symbols

    with open(symbols_file, "r", encoding="utf-8") as handle:
        return [line.strip().upper() for line in handle if line.strip()]


class NSEDownloader:
    def __init__(
        self,
        output_dir: str | Path,
        sleep_seconds: float = 1.0,
        max_reports_per_symbol: int | None = None,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.sleep_seconds = sleep_seconds
        self.max_reports_per_symbol = max_reports_per_symbol

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)
        session.get(BASE_URL, timeout=20)
        time.sleep(self.sleep_seconds)
        return session

    def download_for_symbols(self, symbols: list[str]) -> list[dict]:
        session = self._create_session()
        download_manifest: list[dict] = []

        for symbol in tqdm(symbols, desc="Downloading NSE reports", unit="symbol"):
            symbol_manifest = self._download_symbol(session, symbol)
            download_manifest.append(symbol_manifest)

        manifest_path = self.output_dir / "download_manifest.json"
        write_json(manifest_path, download_manifest)
        return download_manifest

    def _download_symbol(self, session: requests.Session, symbol: str) -> dict:
        symbol_dir = ensure_dir(self.output_dir / symbol)
        api_url = f"{BASE_URL}/api/annual-reports?index=cm&symbol={symbol}"
        response = session.get(api_url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", [])
        rows = rows[: self.max_reports_per_symbol] if self.max_reports_per_symbol else rows

        manifest = {"symbol": symbol, "reports": []}
        for row in tqdm(rows, desc=f"{symbol} reports", unit="report", leave=False):
            file_url = row.get("fileName")
            if not file_url:
                continue

            parsed = urlparse(file_url)
            filename = Path(parsed.path).name or f"{symbol}_report.pdf"
            output_path = symbol_dir / filename

            report_entry = {
                "symbol": symbol,
                "source_url": file_url,
                "filename": filename,
                "saved_path": str(output_path),
                "metadata": {
                    key: normalize_whitespace(str(value))
                    for key, value in row.items()
                    if value is not None
                },
            }

            if not output_path.exists():
                try:
                    download_response = session.get(file_url, timeout=90)
                    download_response.raise_for_status()
                    with open(output_path, "wb") as handle:
                        handle.write(download_response.content)
                    time.sleep(self.sleep_seconds)
                except requests.HTTPError as exc:
                    status_code = exc.response.status_code if exc.response is not None else "unknown"
                    print(f"[WARN] Skipping missing/unavailable NSE file ({status_code}) for {symbol}: {file_url}")
                    report_entry["download_error"] = f"HTTP {status_code}"
                    manifest["reports"].append(report_entry)
                    continue
                except requests.RequestException as exc:
                    print(f"[WARN] Skipping failed NSE download for {symbol}: {file_url}")
                    report_entry["download_error"] = str(exc)
                    manifest["reports"].append(report_entry)
                    continue

            extracted_files = self._extract_if_zip(output_path, symbol_dir)
            if extracted_files:
                report_entry["extracted_files"] = [str(path) for path in extracted_files]

            manifest["reports"].append(report_entry)

        return manifest

    def _extract_if_zip(self, archive_path: Path, output_dir: Path) -> list[Path]:
        if archive_path.suffix.lower() != ".zip":
            return []

        extracted: list[Path] = []
        if not zipfile.is_zipfile(archive_path):
            print(f"[WARN] Skipping invalid ZIP returned by NSE: {archive_path.name}")
            return []

        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    archive.extract(member, output_dir)
                    extracted.append(output_dir / member)
            archive_path.unlink(missing_ok=True)
        except zipfile.BadZipFile:
            print(f"[WARN] Bad ZIP file from NSE, kept as-is: {archive_path.name}")
            return []

        return extracted
