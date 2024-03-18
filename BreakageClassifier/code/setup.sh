if [ -f ~/.mozilla/managed-storage/uBlock0@raymondhill.net.json ]
then 
    echo "extension file exists"
    exit 0
else 
    echo "creating directory (if not exist) and copying file"
    mkdir ~/.mozilla/managed-storage -p
    cp ./uBlock0@raymondhill.net.json ~/.mozilla/managed-storage
fi
echo "setup complete"
