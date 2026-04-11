"""Discover fleet repos via GitHub API."""
import json
import os
import urllib.request

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
API = "https://api.github.com"


def discover_fleet_repos(owner="SuperInstance", per_page=100):
    """Discover all repos in the fleet."""
    url = f"{API}/users/{owner}/repos?per_page={per_page}&sort=updated"
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        repos = json.loads(resp.read().decode())
    return [r for r in repos if not r.get("fork")]


def find_vessel_repos(owner="SuperInstance"):
    """Find vessel repos (agent embodiments)."""
    repos = discover_fleet_repos(owner)
    return [r for r in repos if r["name"].endswith("-vessel")]


def find_task_repos(owner="SuperInstance"):
    """Find repos with message-in-a-bottle TASKS.md."""
    repos = discover_fleet_repos(owner)
    task_repos = []
    for r in repos[:20]:
        try:
            url = f"{API}/repos/{owner}/{r['name']}/contents/message-in-a-bottle/TASKS.md"
            req = urllib.request.Request(url, headers={
                "Authorization": f"token {GITHUB_TOKEN}",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                task_repos.append(r)
        except:
            pass
    return task_repos
