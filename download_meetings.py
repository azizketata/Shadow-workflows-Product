"""Download U.S. city council meeting videos and agendas.

Uses yt-dlp for YouTube video downloads and the Legistar public API
for structured agenda retrieval. Organizes output into per-meeting
folders for the Meeting Process Twin pipeline.

Usage:
    python download_meetings.py --count 5 --cities seattle denver
    python download_meetings.py --dry-run --cities seattle
    python download_meetings.py --skip-video --cities seattle denver
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from dateutil import parser as dateutil_parser


# ============================================================
# City configuration
# ============================================================

CITIES = {
    "seattle": {
        "display_name": "Seattle",
        "youtube_search_query": "Seattle City Council full meeting",
        "youtube_channel_filter": "Seattle Channel",
        "legistar_client": "seattle",
        "legistar_body": "City Council",
    },
    "denver": {
        "display_name": "Denver",
        "youtube_search_query": "Denver City Council meeting",
        "youtube_channel_filter": None,  # various uploaders
        "legistar_client": "denver",
        "legistar_body": "City Council",
    },
    "boston": {
        "display_name": "Boston",
        "youtube_search_query": "Boston City Council meeting",
        "youtube_channel_filter": "Boston City Council",
        "legistar_client": "boston",
        "legistar_body": "City Council",
    },
    "alameda": {
        "display_name": "Alameda",
        "youtube_search_query": "Alameda City Council meeting",
        "youtube_channel_filter": "City of Alameda",
        "legistar_client": "alameda",
        "legistar_body": "City Council",
    },
}

LEGISTAR_BASE = "https://webapi.legistar.com/v1"


# ============================================================
# Utility helpers
# ============================================================

def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def sanitize_folder_name(name: str) -> str:
    """Remove characters invalid in Windows folder names."""
    return re.sub(r'[<>:"/\\|?*]', '', name).strip()


def check_dependencies(skip_video: bool) -> bool:
    """Verify yt-dlp is available."""
    if skip_video:
        return True
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"  yt-dlp version: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
    return False


# ============================================================
# YouTube functions
# ============================================================

def search_youtube_videos(
    search_query: str,
    channel_filter: Optional[str],
    max_results: int = 20,
) -> list[dict]:
    """Search YouTube for council meeting videos using ytsearch."""
    search_url = f"ytsearch{max_results * 2}:{search_query} 2026"
    print(f"  Searching YouTube: \"{search_query} 2026\" ...")

    try:
        result = subprocess.run(
            [
                "yt-dlp", "--flat-playlist",
                "--print", "%(id)s\t%(title)s\t%(channel)s\t%(duration_string)s",
                search_url,
            ],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("  WARNING: yt-dlp search timed out")
        return []

    if result.returncode != 0:
        print(f"  WARNING: yt-dlp search failed: {result.stderr[:200]}")
        return []

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        vid_id, title, channel, duration = parts[0], parts[1], parts[2], parts[3]

        # Filter by channel if specified
        if channel_filter and channel_filter.lower() not in channel.lower():
            continue

        # Filter out short videos (likely clips/summaries, not full meetings)
        # Full council meetings are typically 30min+
        try:
            dur_parts = duration.split(":")
            if len(dur_parts) == 2:
                minutes = int(dur_parts[0])
            elif len(dur_parts) == 3:
                minutes = int(dur_parts[0]) * 60 + int(dur_parts[1])
            else:
                minutes = 0
            if minutes < 25:
                continue
        except (ValueError, IndexError):
            pass

        videos.append({
            "id": vid_id,
            "title": title,
            "channel": channel,
            "duration": duration,
            "upload_date": "",  # Not available from ytsearch
            "url": f"https://www.youtube.com/watch?v={vid_id}",
        })

    return videos[:max_results]


def parse_date_from_title(title: str, upload_date: str = "") -> Optional[date]:
    """Extract meeting date from a YouTube video title."""
    # Strategy 1: dateutil fuzzy parsing
    try:
        dt = dateutil_parser.parse(title, fuzzy=True)
        if 2020 <= dt.year <= 2030:
            return dt.date()
    except (ValueError, OverflowError):
        pass

    # Strategy 2: regex for MM/DD/YYYY or M/D/YYYY
    m = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', title)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        try:
            return date(year, month, day)
        except ValueError:
            pass

    # Strategy 3: regex for "Month DD, YYYY"
    m = re.search(r'(\w+)\s+(\d{1,2}),?\s*(\d{4})', title)
    if m:
        try:
            dt = dateutil_parser.parse(m.group(0))
            return dt.date()
        except (ValueError, OverflowError):
            pass

    # Strategy 4: fallback to yt-dlp upload_date (YYYYMMDD)
    if upload_date and len(upload_date) == 8:
        try:
            return datetime.strptime(upload_date, "%Y%m%d").date()
        except ValueError:
            pass

    return None


def download_video(
    video_url: str,
    output_path: str,
    max_height: int = 720,
) -> bool:
    """Download a YouTube video using yt-dlp."""
    print(f"  Downloading video to {os.path.basename(output_path)} ...")
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
                "--merge-output-format", "mp4",
                "--no-playlist",
                "-o", output_path,
                video_url,
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            return True
        print(f"  WARNING: yt-dlp error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("  WARNING: Video download timed out (10 min limit)")
    except Exception as e:
        print(f"  WARNING: Download failed: {e}")
    return False


# ============================================================
# Legistar API functions
# ============================================================

def fetch_legistar_events(
    client: str,
    body_name: str,
    since_date: date,
    max_events: int = 20,
) -> list[dict]:
    """Fetch council meeting events from the Legistar API."""
    since_str = since_date.strftime("%Y-%m-%d")
    url = (
        f"{LEGISTAR_BASE}/{client}/events"
        f"?$filter=EventBodyName eq '{body_name}' "
        f"and EventDate ge datetime'{since_str}'"
        f"&$orderby=EventDate desc"
        f"&$top={max_events}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  WARNING: Legistar API error: {e}")
        return []


def fetch_event_items(client: str, event_id: int) -> list[dict]:
    """Fetch agenda items for a specific Legistar event."""
    url = (
        f"{LEGISTAR_BASE}/{client}/events/{event_id}/eventitems"
        f"?$orderby=EventItemAgendaSequence"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  WARNING: Legistar EventItems error: {e}")
        return []


def format_agenda(
    items: list[dict],
    city_name: str,
    event_date: date,
) -> str:
    """Format Legistar EventItems into a numbered agenda.txt."""
    header = (
        f"{city_name.upper()} CITY COUNCIL REGULAR MEETING AGENDA\n"
        f"{event_date.strftime('%B %Y')}\n\n"
    )

    # Filter to items with meaningful agenda titles
    titles = []
    for item in items:
        title = (item.get("EventItemTitle") or "").strip()
        if not title:
            continue
        # Skip metadata-only entries
        if re.match(r'^(January|February|March|April|May|June|July|August|'
                     r'September|October|November|December)\s+\d', title):
            continue
        if title.lower() in ("", "page break", "---"):
            continue
        # Skip entries that are mostly underscores/dashes (signature lines)
        if re.match(r'^[_\-\s]{5,}', title):
            continue
        # Skip signed-by / clerk lines
        if re.match(r'^(Signed by|Approved by|\w+ \w+, (Deputy )?City Clerk)', title, re.IGNORECASE):
            continue
        # Skip council president signature lines
        if re.search(r'Council President|City Clerk', title, re.IGNORECASE) and len(title) < 80:
            continue
        # Skip "Journal:" and "Bills:" section headers (keep actual bill descriptions)
        if title.rstrip(':') in ("Journal", "Bills"):
            continue
        # Skip boilerplate instructional text (common in Legistar)
        if any(kw in title.lower() for kw in [
            "how to provide", "how to listen", "how to attend",
            "connecting to the webinar", "webinar id",
            "zoom phone number", "zoom meeting id", "zoom registration",
            "public speaking times", "speakers cannot cede",
            "visuals to be shown", "meeting rules of order",
            "accessible seating", "equipment for the hearing",
            "translators will be available", "closed captioning",
            "sign up to receive", "view documents related",
            "documents related to this agenda",
            "language access services",
            "stream online via", "watch king county",
            "listen to the meeting by telephone",
            "reasonable accommodation",
        ]):
            continue
        # Skip items that are just URLs
        if re.match(r'^https?://', title.strip()):
            continue
        # Skip items that are just meeting logistics paragraphs (>200 chars with no agenda keywords)
        if len(title) > 200 and not any(kw in title.lower() for kw in [
            "ordinance", "resolution", "motion", "hearing", "approval",
            "adoption", "authorize", "recommendation", "amendment",
        ]):
            continue
        # Normalize ALL-CAPS to Title Case
        if title == title.upper() and len(title) > 3:
            title = title.title()
        titles.append(title)

    if not titles:
        return header + "# No agenda items found in Legistar API\n"

    lines = []
    for i, title in enumerate(titles, 1):
        lines.append(f"{i}. {title}")

    return header + "\n".join(lines) + "\n"


def match_video_to_event(
    video_date: date,
    events: list[dict],
) -> Optional[dict]:
    """Find a Legistar event matching a video date (+/- 1 day)."""
    for event in events:
        event_date_str = event.get("EventDate", "")
        try:
            event_date = datetime.fromisoformat(
                event_date_str.replace("T", " ").split(".")[0]
            ).date()
            if abs((video_date - event_date).days) <= 1:
                return event
        except (ValueError, AttributeError):
            continue
    return None


# ============================================================
# Main orchestration
# ============================================================

def process_city(
    city_key: str,
    config: dict,
    output_dir: str,
    count: int,
    since_date: date,
    skip_video: bool = False,
    max_height: int = 720,
    dry_run: bool = False,
) -> dict:
    """Download meetings for a single city."""
    city_name = config["display_name"]
    print_section(f"{city_name}")

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "agendas": 0}

    # Step 1: Search YouTube for meeting videos
    videos = []
    if config.get("youtube_search_query"):
        videos = search_youtube_videos(
            config["youtube_search_query"],
            config.get("youtube_channel_filter"),
            max_results=count * 3,  # fetch extra since we skip videos without agendas
        )
        print(f"  Found {len(videos)} matching videos")
    else:
        print(f"  No YouTube videos configured for {city_name}, skipping")
        return stats

    # Step 2: Fetch Legistar events
    events = []
    if config.get("legistar_client") and config.get("legistar_body"):
        events = fetch_legistar_events(
            config["legistar_client"],
            config["legistar_body"],
            since_date,
            max_events=max(count * 3, 30),
        )
        print(f"  Found {len(events)} Legistar events")

    # Step 3: Match videos to Legistar agendas — only keep pairs
    for video in videos:
        video_date = parse_date_from_title(
            video["title"], video.get("upload_date", "")
        )
        if not video_date:
            print(f"  SKIP: Could not parse date from: {video['title']}")
            stats["failed"] += 1
            continue

        # Require a matching Legistar agenda — skip if none
        event = match_video_to_event(video_date, events) if events else None
        if not event:
            print(f"  SKIP: No agenda for {video_date} — {video['title']}")
            stats["skipped"] += 1
            continue

        date_str = video_date.strftime("%Y-%m-%d")
        folder_name = f"{city_key}_{date_str}"
        folder_path = os.path.join(output_dir, folder_name)

        # Check if already downloaded
        video_path = os.path.join(folder_path, "video.mp4")
        agenda_path = os.path.join(folder_path, "agenda.txt")

        if os.path.exists(video_path) and os.path.exists(agenda_path):
            print(f"  SKIP: {folder_name} already exists")
            stats["skipped"] += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] Would download: {folder_name}")
            print(f"    Video: {video['title']}")
            print(f"    Agenda: Legistar (event {event.get('EventId')})")
            stats["downloaded"] += 1
            continue

        # Create folder
        os.makedirs(folder_path, exist_ok=True)

        # Download agenda first
        event_id = event.get("EventId")
        items = fetch_event_items(config["legistar_client"], event_id)
        agenda_text = format_agenda(items, city_name, video_date)
        with open(agenda_path, "w", encoding="utf-8") as f:
            f.write(agenda_text)
        print(f"  Agenda: {len(items)} items from Legistar")
        stats["agendas"] += 1

        # Download video
        if not skip_video:
            success = download_video(video["url"], video_path, max_height)
            if not success:
                print(f"  FAILED: Could not download video for {folder_name}")
                stats["failed"] += 1
        else:
            print(f"  SKIP VIDEO: {folder_name} (--skip-video)")

        # Write metadata
        metadata = {
            "city": city_name,
            "city_key": city_key,
            "date": date_str,
            "video_title": video["title"],
            "video_url": video["url"],
            "youtube_id": video["id"],
            "legistar_event_id": event.get("EventId") if event else None,
            "legistar_agenda_pdf": event.get("EventAgendaFile") if event else None,
            "downloaded_at": datetime.now().isoformat(),
        }
        with open(os.path.join(folder_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        stats["downloaded"] += 1
        print(f"  OK: {folder_name}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download U.S. city council meeting videos and agendas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Number of most recent meetings to download per city",
    )
    parser.add_argument(
        "--cities", nargs="+", default=["seattle", "denver", "boston", "alameda"],
        choices=list(CITIES.keys()),
        help="Cities to download from",
    )
    parser.add_argument(
        "--output-dir", default="meetings",
        help="Base output directory",
    )
    parser.add_argument(
        "--since", default=None,
        help="Only fetch meetings after this date (YYYY-MM-DD). Default: 6 months ago",
    )
    parser.add_argument(
        "--skip-video", action="store_true",
        help="Download agendas only, skip video downloads",
    )
    parser.add_argument(
        "--max-height", type=int, default=720,
        help="Maximum video height in pixels",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be downloaded without actually downloading",
    )

    args = parser.parse_args()

    print_section("Meeting Downloader")
    print(f"  Cities:    {', '.join(args.cities)}")
    print(f"  Count:     {args.count} per city")
    print(f"  Output:    {args.output_dir}")
    print(f"  Dry run:   {args.dry_run}")
    print(f"  Skip video: {args.skip_video}")

    # Check dependencies
    if not check_dependencies(args.skip_video or args.dry_run):
        sys.exit(1)

    # Parse since date
    if args.since:
        since_date = datetime.strptime(args.since, "%Y-%m-%d").date()
    else:
        since_date = date.today() - timedelta(days=180)
    print(f"  Since:     {since_date}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each city
    all_stats = {}
    for city_key in args.cities:
        if city_key not in CITIES:
            print(f"  WARNING: Unknown city '{city_key}', skipping")
            continue
        stats = process_city(
            city_key=city_key,
            config=CITIES[city_key],
            output_dir=args.output_dir,
            count=args.count,
            since_date=since_date,
            skip_video=args.skip_video,
            max_height=args.max_height,
            dry_run=args.dry_run,
        )
        all_stats[city_key] = stats

    # Print summary
    print_section("Summary")
    total_downloaded = 0
    total_agendas = 0
    for city_key, stats in all_stats.items():
        city_name = CITIES[city_key]["display_name"]
        print(f"  {city_name}: {stats['downloaded']} downloaded, "
              f"{stats['agendas']} agendas, "
              f"{stats['skipped']} skipped, "
              f"{stats['failed']} failed")
        total_downloaded += stats["downloaded"]
        total_agendas += stats["agendas"]

    print(f"\n  Total: {total_downloaded} meetings, {total_agendas} agendas")
    print(f"  Output: {os.path.abspath(args.output_dir)}")

    if args.dry_run:
        print("\n  (Dry run — nothing was actually downloaded)")

    if not args.dry_run and not args.skip_video:
        print(f"\n  Next step: Run the pipeline on a downloaded meeting:")
        print(f"    python test_pipeline.py --api-key sk-... \\")
        print(f"        --video meetings/<city_date>/video.mp4 \\")
        print(f"        --agenda meetings/<city_date>/agenda.txt \\")
        print(f"        --output-dir meetings/<city_date>/results")


if __name__ == "__main__":
    main()
