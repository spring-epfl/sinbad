# Scraping Forums

An essential part of training the pipeline is to scrape issues forums for broken sites.

## Available Forums

We mainly focus on the following two forums to scrape breakage issues:

1. [Ublock Origin Github Issues](https://github.com/uBlockOrigin/uBlock-issues/issues)
2. [EasyList forum](https://forums.lanik.us/viewforum.php?f=64-report-incorrectly-removed-content)

---

## Dataset structure

The forum dataset is divided into two parts:

- CSV dataset file: ex: `./ublock/data/ublock-data.csv`
- Directory `[...]/data/filterlists/`

### CSV D ataset

The important features are:
| Feature | Type | Description|
| ----------- | ----------- | ----|
| `id` | String |The issue identifier|
| `title` | String |The title of the forum issue|
| `created_at` | Datetime `YYYY-MM-DDThh:mm:ssZ`| |
| `repo` | String | The repository path on github |
| `before_commit`| String | The commit Id before the first fix was applied|
| `after_commit` | String | The commit Id after the last fix was applied |
| `directories` | Stringified List | The list of changed directories in the fixing commit |
| `label` | Char | "S": static, "D": dynamic, "X": not reproducible, "N": no breakage|
| `test_url` | URL | The test URL to crawl|

### Filterlists directory

The `filterlists/` directory contains subdirectories named with the issue `id`. inside we have two files `before.txt` and `after.txt`.

> - `filterlists/`
>   - `<issue-id>`
>     - `before.txt`: the compiled filterlist before the first fix.
>     - `after.txt`: the compiled filterlist after the last fix.

---

## Procedure

Scraping a forum varies depending on the structure of the forum (Easylist vs. Ublock origin). Hence each new forum has different rules:

### Steps for Ublock Origin `./ublock`

We have the scraped issues in `data_uassets.json` and labeled the issues `manual_data.json`.

1. Compile the dataset.

   ```
   python builder.py --out [output folder path]
   ```

2. Extract test URLs. this process is semi-automatic, if the test URL is unclear from the post, you will be shown the post and have a prompt to manually enter the URL or dismiss the Post as incomplete.

   ```
   python extract_urls.py --out [output folder path]
   ```

3. Combine the dataset and test URLs.

   ```
   python extract_urls.py --action combine --out [output folder path]
   ```

4. Get Github token to access API by following [this](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) documentation. Once you have your token, you need to store it somewhere secure (for example, a password manager). Once you are ready to download the filterlists, set a new environment variable `GITHUB_TOKEN` with the value.

```
export GITHUB_TOKEN="[your token]"
```

5. Download the filterlists for the issues in the dataset. In the `/forums` directory:
   ```
   python filterlists.py --input [path to CSV dataset] --out [filterlist outputs]
   ```

### Steps for Easylist Forum `./easylist`

Todo: add from the forums.ipynb

### Manual Labeling

Labeling the breakage-type and elements

```
python manual_label.py -i <dataset.csv path> -a breakage-type element
```

Labeling the reproducibility

```
python manual_label.py -i <dataset.csv path> -a reproduce -f <filterlists/ directory path>
```
