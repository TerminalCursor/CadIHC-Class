from ..tools.call_graph import cg
from typing import Self, Any
from paramiko import SSHClient, AutoAddPolicy
from tqdm.auto import tqdm


def progress_bar(pbar: Any, transfer: int, toTransfer: int) -> None:
    """# Progress bar for transfers"""
    d = transfer / toTransfer * 100
    pbar.update(d - pbar.n)


class RemoteFile:
    """# Class to pull remote files"""

    remote_filename: str
    local_filename: str

    hostname: str
    username: str | None
    key_filename: str | None

    def __init__(
        self: Self,
        remote_filename: str,
        hostname: str,
        local_filename: str,
        username: str | None = None,
        key_filename: str | None = None,
    ):
        cg.add("RemoteFile.__init__")
        self.remote_filename = remote_filename
        self.local_filename = local_filename
        self.hostname = hostname
        self.username = username
        self.key_filename = key_filename

    def download(self: Self) -> bool:
        """# Attempt to download remote resource"""
        cg.add("RemoteFile.download")

        success = False
        try:
            ssh = SSHClient()
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            kwargs = {}
            if self.username is not None:
                kwargs["username"] = self.username
            if self.key_filename is not None:
                kwargs["key_filename"] = self.key_filename
            ssh.connect(self.hostname, **kwargs)  # pyright: ignore
            with ssh.open_sftp() as sftp:
                pbar = tqdm(total=100)
                pbar.set_description(f"{self.remote_filename}")
                progress = lambda x, y: progress_bar(pbar, x, y)  # pyright: ignore
                sftp.get(
                    self.remote_filename,
                    self.local_filename,
                    callback=progress,  # pyright: ignore
                )
                pbar.close()
            ssh.close()
            success = True
        except Exception as e:
            print("RemoteResource.download failed!")
            print(e)
        return success

    def upload(self: Self) -> bool:
        """# Attempt to upload remote resource"""
        cg.add("RemoteFile.upload")

        success = False
        try:
            ssh = SSHClient()
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            kwargs = {}
            if self.username is not None:
                kwargs["username"] = self.username
            if self.key_filename is not None:
                kwargs["key_filename"] = self.key_filename
            ssh.connect(self.hostname, **kwargs)  # pyright: ignore
            with ssh.open_sftp() as sftp:
                remote_path = "/".join(self.remote_filename.split("/")[:-1])
                if len(remote_path) > 0:
                    try:
                        sftp.mkdir(remote_path)
                    except Exception as e:
                        pass
                pbar = tqdm(total=100)
                pbar.set_description(f"{self.local_filename}")
                progress = lambda x, y: progress_bar(pbar, x, y)  # pyright: ignore
                sftp.put(
                    self.local_filename,
                    self.remote_filename,
                    callback=progress,  # pyright: ignore
                )
                pbar.close()
            ssh.close()
            success = True
        except Exception as e:
            print("RemoteResource.upload failed!")
            print(e)
        return success
