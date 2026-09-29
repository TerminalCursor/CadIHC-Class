from histclass.interfaces.remote_file import RemoteFile
from histclass.tools.env import get_env

hostname = get_env("REMOTE_HOSTNAME").unwrap()
username = get_env("REMOTE_UNAME").data
keyfile = get_env("REMOTE_KEY_FILE").data
print(hostname, username, keyfile)

rf = RemoteFile(
    remote_filename=".bash_history",
    hostname=hostname,
    local_filename="tests/outputs/history",
    username=username,
    key_filename=keyfile,
)

rf.download()

rf = RemoteFile(
    remote_filename="test_file.toml",
    hostname=hostname,
    local_filename="pyproject.toml",
    username=username,
    key_filename=keyfile,
)

rf.upload()
