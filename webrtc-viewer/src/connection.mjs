// Keep the signaling endpoint local even if connection metadata is malformed.
// NVIDIA's media override replaces the server's loopback ICE address and port.
export function streamConnection(value) {
  if (!value || value.signalingServer !== "127.0.0.1") {
    throw new Error("Signaling must use the local SSH tunnel.");
  }
  const validPort = (port) => Number.isInteger(port) && port > 0 && port <= 65535;
  const octets = typeof value.mediaServer === "string" ? value.mediaServer.split(".") : [];
  if (octets.length !== 4 || octets.some((v) => !/^\d{1,3}$/.test(v) || Number(v) > 255)
      || !validPort(value.mediaPort) || !validPort(value.signalingPort)) {
    throw new Error("Invalid media address or streaming port. Restart the viewer command.");
  }
  return {
    signalingServer: "127.0.0.1",
    signalingPort: value.signalingPort,
    mediaServer: value.mediaServer,
    mediaPort: value.mediaPort,
    forceWSS: false,
  };
}
