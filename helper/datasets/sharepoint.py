"""Minimal client for anonymous SharePoint / OneDrive share links."""

import http.cookiejar
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm


class SharedFolder:
    """Lists and downloads files under a public 'anyone with the link' folder."""

    def __init__(self, share_url: str):
        jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        with self._opener.open(share_url) as resp:
            landing = urllib.parse.urlparse(resp.geturl())
        # Landing URL looks like /personal/<user>/_layouts/15/onedrive.aspx?id=<root path>
        self.root = urllib.parse.parse_qs(landing.query)["id"][0]
        site = "/".join(self.root.split("/")[:3])
        self._api = f"{landing.scheme}://{landing.netloc}{site}/_api/web"

    def _url(self, rel: str, suffix: str) -> str:
        path = urllib.parse.quote(f"{self.root}/{rel}".rstrip("/"))
        kind = "GetFileByServerRelativeUrl" if suffix == "$value" else "GetFolderByServerRelativeUrl"
        return f"{self._api}/{kind}('{path}')/{suffix}"

    def _list(self, rel: str, kind: str) -> dict:
        req = urllib.request.Request(self._url(rel, kind), headers={"Accept": "application/json;odata=nometadata"})
        with self._opener.open(req) as resp:
            return {x["Name"]: int(x.get("Length", 0)) for x in json.load(resp)["value"]}

    def files(self, rel: str = "") -> dict:
        """Returns {name: size_bytes} for files directly under `rel`."""
        return self._list(rel, "Files")

    def folders(self, rel: str = "") -> list:
        return sorted(self._list(rel, "Folders"))

    def download(self, rel: str, dst: Path, size: int | None = None) -> Path:
        """Downloads `rel` to `dst`, skipping it when a file of the expected size exists."""
        dst = Path(dst)
        if dst.exists() and size is not None and dst.stat().st_size == size:
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with self._opener.open(self._url(rel, "$value")) as resp, open(tmp, "wb") as out:
            total = size or int(resp.headers.get("Content-Length") or 0) or None
            with tqdm(total=total, unit="B", unit_scale=True, desc=dst.name, leave=False) as bar:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    bar.update(len(chunk))
        if total and tmp.stat().st_size != total:  # the server can close the connection early
            raise IOError(f"{dst.name}: got {tmp.stat().st_size} of {total} bytes")
        shutil.move(tmp, dst)
        return dst
