from typing import Self, Any
from paramiko import SSHClient, AutoAddPolicy
from tqdm.auto import tqdm


def progress_bar(bar_class: Any, transfer: int, toTransfer: int) -> None:
    """
    Updates the progress bar by the percentage of transfer done.

    Args:
        bar_class (Any): The progress bar object.
        transfer (int): The number of bytes transferred so far.
        toTransfer (int): The total number of bytes to be transferred.

    Returns:
        None
    """
    transferred_p = transfer / toTransfer * 100
    bar_class.update(transferred_p - bar_class.n)


class RemoteFile:
    """
    A class to pull remote files from a server using SSH.

    Attributes:
        remote_filename (str): The filename of the remote file.
        local_filename (str): The local filename where the file will be saved.
        hostname (str): The hostname or IP address of the server.
        username (str | None): The username to use for authentication.
        key_filename (str | None): The path to the SSH private key.

    Methods:
        download(): Attempts to download a remote resource.
        upload(): Attempts to upload a local resource.
    """

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
        """
        Initializes the RemoteFile object.

        Args:
            remote_filename (str): The filename of the remote file.
            hostname (str): The hostname or IP address of the server.
            local_filename (str): The local filename where the file will be saved.
            username (str | None, optional): The username to use for authentication. Defaults to None.
            key_filename (str | None, optional): The path to the SSH private key. Defaults to None.

        Returns:
            None
        """
        self.remote_filename = remote_filename
        self.local_filename = local_filename
        self.hostname = hostname
        self.username = username
        self.key_filename = key_filename

    def download(self: Self) -> bool:
        """
        Attempts to download a remote resource.

        Returns:
            bool: True if the download is successful, False otherwise.
        """
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
        """
        Attempts to upload a local resource.

        Returns:
            bool: True if the upload is successful, False otherwise.
        """
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
