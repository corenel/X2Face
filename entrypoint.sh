#!/usr/bin/env bash

COLOR_BLUE='\033[0;34m'
COLOR_NULL='\033[0m'
USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

echo -e "${COLOR_BLUE}Starting SSH Server${COLOR_NULL}"
chmod 700 /home/user/.ssh
chmod 600 /home/user/.ssh/authorized_keys
chmod g-w,o-w /home/user
/usr/sbin/sshd -p 22222

echo -e "${COLOR_BLUE}Starting with UID: ${USER_ID}, GID: ${GROUP_ID}${COLOR_NULL}"
usermod -u ${USER_ID} user
groupmod -g ${GROUP_ID} user
export HOME=/home/user

# install missing packages
apt install -y rsync
pip install torchvision==0.2.0

echo -e "${COLOR_BLUE}Executing command${COLOR_NULL}"
exec /usr/sbin/gosu user "$@"
