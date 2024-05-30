import subprocess


url_prefix = (
    "https://s3.us-west-1.wasabisys.com/predictive-coding-recovers-maps/weights/"
)
with open("files.txt", "r") as f:
    for line in f:
        url = url_prefix + line.strip()
        subprocess.run(["curl", "-LO", url])
