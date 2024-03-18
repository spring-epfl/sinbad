from datetime import datetime
import os

from datetime import datetime
from io import TextIOWrapper
import os
from typing import Optional
import requests

AUTH_TOKEN = os.environ["GITHUB_TOKEN"]
headers = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": "token " + AUTH_TOKEN,
}

URL_REGEX = r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))'

__dir__ = os.path.dirname(os.path.abspath(__file__))


class BadFilterlistSource(Exception):
    pass


def get_commit_date(
    owner_repo: str,
    commit_sha: str,
):
    url = f"https://api.github.com/repos/{owner_repo}/commits/{commit_sha}"

    dt_str = requests.get(
        url,
        headers=headers,
    ).json()["commit"][
        "committer"
    ]["date"]

    # example string '2022-03-18T11:20:48Z'
    return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")


def get_commit_parent(
    owner_repo: str,
    commit_sha: str,
):
    url = f"https://api.github.com/repos/{owner_repo}/commits/{commit_sha}"

    return requests.get(
        url,
        headers=headers,
    ).json()["parents"][
        0
    ]["sha"]


def get_commit_files(
    owner_repo: str,
    commit_sha: str,
):
    url = f"https://api.github.com/repos/{owner_repo}/commits/{commit_sha}"

    resp = requests.get(
        url,
        headers=headers,
    ).json()

    if "files" not in resp:
        raise Exception(f"Error requesting: {url}")

    return resp["files"]


def get_latest_commit_until(
    owner_repo: str,
    until: datetime,
):
    url = f"https://api.github.com/repos/{owner_repo}/commits?until={until.isoformat()}Z&per_page=1"

    commits = requests.get(
        url,
        headers=headers,
    ).json()

    if len(commits) == 0:
        return None

    return {
        "sha": commits[0]["sha"],
        "date": commits[0]["commit"]["committer"]["date"],
    }


def dump_static_filterlist(stream: TextIOWrapper, url: str):
    try:
        resp = requests.get(url)
    except requests.exceptions.ContentDecodingError as e:
        print(e)
        raise BadFilterlistSource(f"Error requesting: {url}")

    st_lines = stream.tell()

    if resp.status_code != 200:
        raise Exception(f"Error requesting: {url}, status code: {resp.status_code}")

    stream.write(resp.text)

    return stream.tell() - st_lines


def get_repo_and_commit_from_url(url):
    if "api.github.com" in url:
        commits_sep = "/commits/"
        repo_sep = "https://api.github.com/repos/"

    else:
        commits_sep = "/commit/"
        repo_sep = "https://github.com/"

    repo_url, commit_id = url.split(commits_sep)
    _, repo_url = repo_url.split(repo_sep)

    return repo_url, commit_id


def dump_list_from_commit(
    stream: TextIOWrapper,
    owner_repo: str,
    commit_sha: str,
    dirs: Optional[list[str]] = None,
    files: Optional[list[str]] = None,
):
    n_lines = stream.tell()

    stream.write(
        f"""
# Github repository: {owner_repo}
# Commit SHA: {commit_sha}

"""
    )

    for dir in dirs:
        # get the files in the directory
        url = (
            f"https://api.github.com/repos/{owner_repo}/contents/{dir}?ref={commit_sha}"
        )

        files_in_dir = requests.get(
            url,
            headers=headers,
        ).json()

        if "message" in files_in_dir:
            if files_in_dir["message"] == "Not Found":
                print(
                    f"Directory not found: https://api.github.com/repos/{owner_repo}/contents/{dir}?ref={commit_sha}"
                )
                stream.write(f"# ------ {dir} not found ------\n")
                continue
            else:
                raise Exception(
                    f"Error requesting: https://api.github.com/repos/{owner_repo}/contents/{dir}?ref={commit_sha}"
                )

        try:
            for file in files_in_dir:
                stream.write(f"# ------ {file['name']} ------\n")

                text = requests.get(
                    file["download_url"],
                    headers=headers,
                ).text

                for line in text.splitlines(keepends=True):
                    if line.startswith("! ") or line.startswith("# "):
                        continue
                    stream.write(line)

        except Exception as e:
            print(files_in_dir)
            raise e

    # files are file paths
    for file in files:
        resp = requests.get(
            f"https://raw.githubusercontent.com/{owner_repo}/{commit_sha}/{file}",
            # headers=headers,
        )
        if resp.status_code != 200:
            if "404: Not Found" in resp.text:
                print(resp.text)
                print(
                    f"File not found: https://raw.githubusercontent.com/{owner_repo}/{commit_sha}/{file}"
                )
                stream.write(f"# ------ {file} not found ------\n")
            else:
                raise Exception(
                    f"Error requesting: https://raw.githubusercontent.com/{owner_repo}/{commit_sha}/{file}"
                )

        stream.write(f"# ------ {file} ------\n")

        for line in resp.text.splitlines(keepends=True):
            if line.startswith("! ") or line.startswith("# "):
                continue
            stream.write(line)

    return stream.tell() - n_lines
