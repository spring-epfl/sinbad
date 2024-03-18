sudo apt install ffmpeg
pip install -r requirements.txt;
bash install-firefox.sh
apt-get update;
apt-get install -y wget bzip2 libxtst6 libgtk-3-0 libx11-xcb-dev libdbus-glib-1-2 libxt6 libpci-dev libasound2;
rm -rf /var/lib/apt/lists/*
mkdir -p drivers
wget https://github.com/mozilla/geckodriver/releases/download/v0.31.0/geckodriver-v0.31.0-linux64.tar.gz
tar -xf geckodriver-v0.31.0-linux64.tar.gz -C ./drivers
rm geckodriver-v0.31.0-linux64.tar.gz