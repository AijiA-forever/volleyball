# -*- coding: utf-8 -*-
"""生成本地 CA 与服务器证书（用于局域网 HTTPS，解决浏览器禁用摄像头的问题）。"""
import ipaddress
import socket
import sys
from datetime import datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).resolve().parent
CERT_DIR = ROOT / "certs"
CERT_DIR.mkdir(parents=True, exist_ok=True)


def local_ips():
    ips = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            addr = info[4][0]
            try:
                ip = ipaddress.ip_address(addr)
                if ip.version == 4 and not ip.is_loopback and not ip.is_link_local:
                    ips.add(addr)
            except ValueError:
                pass
    except Exception:
        pass
    for extra in sys.argv[1:]:
        try:
            ipaddress.ip_address(extra)
            ips.add(extra)
        except ValueError:
            pass
    ips.add("127.0.0.1")
    return sorted(ips)


def main():
    now = datetime.utcnow()
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Volleyball Teaching Local CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    ips = local_ips()
    san = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "volleyball-system")]))
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    (CERT_DIR / "ca.key").write_bytes(
        ca_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption())
    )
    (CERT_DIR / "ca.crt").write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    (CERT_DIR / "server.key").write_bytes(
        server_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption())
    )
    (CERT_DIR / "server.crt").write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    print("证书已生成，覆盖的地址：", ", ".join(ips))
    print("CA 证书：", CERT_DIR / "ca.crt")
    print("服务器证书：", CERT_DIR / "server.crt")


if __name__ == "__main__":
    main()