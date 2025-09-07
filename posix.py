# Mock posix module for Windows compatibility
# This provides basic posix functions that some packages expect on Unix systems

import os

# Basic posix constants and functions that might be expected
O_RDONLY = os.O_RDONLY if hasattr(os, 'O_RDONLY') else 0
O_WRONLY = os.O_WRONLY if hasattr(os, 'O_WRONLY') else 1
O_RDWR = os.O_RDWR if hasattr(os, 'O_RDWR') else 2
O_CREAT = os.O_CREAT if hasattr(os, 'O_CREAT') else 64
O_TRUNC = os.O_TRUNC if hasattr(os, 'O_TRUNC') else 512
O_APPEND = os.O_APPEND if hasattr(os, 'O_APPEND') else 1024

def open(path, flags, mode=0o777):
    return os.open(path, flags, mode)

def close(fd):
    return os.close(fd)

def read(fd, n):
    return os.read(fd, n)

def write(fd, data):
    return os.write(fd, data)

def lseek(fd, pos, how):
    return os.lseek(fd, pos, how)

def unlink(path):
    return os.unlink(path)

def mkdir(path, mode=0o777):
    return os.mkdir(path, mode)

def rmdir(path):
    return os.rmdir(path)

def chdir(path):
    return os.chdir(path)

def getcwd():
    return os.getcwd()

def listdir(path):
    return os.listdir(path)

def stat(path):
    return os.stat(path)

def chmod(path, mode):
    return os.chmod(path, mode)

def chown(path, uid, gid):
    # Windows doesn't support chown in the same way
    pass

def getuid():
    # Return a dummy UID for Windows
    return 1000

def getgid():
    # Return a dummy GID for Windows
    return 1000

def geteuid():
    return getuid()

def getegid():
    return getgid()

def fork():
    raise OSError("fork() not supported on Windows")

def waitpid(pid, options):
    raise OSError("waitpid() not supported on Windows")

def execv(path, args):
    raise OSError("execv() not supported on Windows")

def execvp(file, args):
    raise OSError("execvp() not supported on Windows")

# Signal constants
SIGINT = 2
SIGTERM = 15
SIGHUP = 1

# Path separator
sep = os.sep
pathsep = os.pathsep

def pread(fd, buffersize, offset):
    # Windows doesn't have pread, so we need to simulate it
    current_pos = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, offset, os.SEEK_SET)
    data = os.read(fd, buffersize)
    os.lseek(fd, current_pos, os.SEEK_SET)
    return data

def pwrite(fd, string, offset):
    # Windows doesn't have pwrite, so we need to simulate it
    current_pos = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, offset, os.SEEK_SET)
    result = os.write(fd, string)
    os.lseek(fd, current_pos, os.SEEK_SET)
    return result

def fstat(fd):
    return os.fstat(fd)

def fchmod(fd, mode):
    # Windows may not support fchmod
    pass

def fchown(fd, uid, gid):
    # Windows doesn't support fchown
    pass

def ftruncate(fd, length):
    return os.ftruncate(fd, length)

def fsync(fd):
    return os.fsync(fd)

def fdatasync(fd):
    # Windows doesn't have fdatasync, use fsync
    return os.fsync(fd)

def pipe():
    # Create a pipe using os.pipe
    return os.pipe()

def dup(fd):
    return os.dup(fd)

def dup2(fd, fd2):
    return os.dup2(fd, fd2)

def setsid():
    # Windows doesn't support setsid
    pass

def setpgid(pid, pgrp):
    # Windows doesn't support setpgid
    pass

def getpgid(pid):
    # Return dummy value
    return pid

def tcgetpgrp(fd):
    # Return dummy value
    return 0

def tcsetpgrp(fd, pgrp):
    # Windows doesn't support tcsetpgrp
    pass

def ctermid():
    # Return dummy terminal name
    return "/dev/tty"

def ttyname(fd):
    # Return dummy terminal name
    return "/dev/tty"

def isatty(fd):
    return os.isatty(fd)

def WIFEXITED(status):
    return (status & 0xFF) == 0

def WEXITSTATUS(status):
    return (status >> 8) & 0xFF

def WIFSIGNALED(status):
    return (status & 0x7F) != 0

def WTERMSIG(status):
    return status & 0x7F

def WIFSTOPPED(status):
    return (status & 0xFF) == 0x7F

def WSTOPSIG(status):
    return (status >> 8) & 0xFF

# Additional constants
F_OK = os.F_OK if hasattr(os, 'F_OK') else 0
R_OK = os.R_OK if hasattr(os, 'R_OK') else 4
W_OK = os.W_OK if hasattr(os, 'W_OK') else 2
X_OK = os.X_OK if hasattr(os, 'X_OK') else 1

SEEK_SET = os.SEEK_SET if hasattr(os, 'SEEK_SET') else 0
SEEK_CUR = os.SEEK_CUR if hasattr(os, 'SEEK_CUR') else 1
SEEK_END = os.SEEK_END if hasattr(os, 'SEEK_END') else 2

# File mode constants
S_IFMT = 0o170000
S_IFDIR = 0o040000
S_IFCHR = 0o020000
S_IFBLK = 0o060000
S_IFREG = 0o100000
S_IFIFO = 0o010000
S_IFLNK = 0o120000
S_IFSOCK = 0o140000

S_IRWXU = 0o0700
S_IRUSR = 0o0400
S_IWUSR = 0o0200
S_IXUSR = 0o0100
S_IRWXG = 0o0070
S_IRGRP = 0o0040
S_IWGRP = 0o0020
S_IXGRP = 0o0010
S_IRWXO = 0o0007
S_IROTH = 0o0004
S_IWOTH = 0o0002
S_IXOTH = 0o0001

# Error constants
EEXIST = 17
ENOENT = 2
EACCES = 13
EPERM = 1
