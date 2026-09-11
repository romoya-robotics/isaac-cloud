// Browser client for Isaac's native WebRTC stream. Served from the Isaac
// container on loopback (like noVNC) and reached through the SSH tunnel; the
// SDK module next to this file is downloaded onto the box by isaac_cloud.py.
import { AppStreamer, StreamType, eStatus } from './omniverse-webrtc-streaming-library.js';
import { streamConnection } from './connection.js';

const status = document.querySelector('#status');
const connect = document.querySelector('#connect');
const disconnect = document.querySelector('#disconnect');
const video = document.querySelector('#remote-video');
let requested = false;
let timer;

function report(message) {
  status.textContent = message;
}

function reportEnded(message) {
  clearTimeout(timer);
  report(message);
}

function onEvent(event) {
  console.info('Isaac stream:', event);
  if (event.status === eStatus.error) {
    reportEnded('Connection failed. Check Isaac readiness and UDP access, then disconnect and reconnect.');
  }
}

connect.addEventListener('click', async () => {
  if (requested) return;
  requested = true;
  connect.disabled = true;
  disconnect.disabled = false;
  report('Connecting to Isaac Sim…');
  try {
    // Written by the tunnel command on every (re)connect; never cached.
    const response = await fetch('connection.json', { cache: 'no-store' });
    if (!response.ok) throw new Error('Connection settings are unavailable. Restart the tunnel command.');
    const connection = streamConnection(await response.json());
    timer = setTimeout(() => {
      report('Still waiting for video. Check that Isaac has loaded and your network allows UDP.');
    }, 30000);
    await AppStreamer.connect({
      streamSource: StreamType.DIRECT,
      streamConfig: {
        ...connection,
        videoElementId: 'remote-video',
        audioElementId: 'remote-audio',
        width: 1920,
        height: 1080,
        fps: 60,
        autoLaunch: true,
        authenticate: false,
        maxReconnects: 5,
        nativeTouchEvents: true,
        onUpdate: onEvent,
        onStart: onEvent,
        onStop: () => reportEnded('Stream stopped. Disconnect and reconnect to try again.'),
        onTerminate: () => reportEnded('Stream ended. Disconnect and reconnect to try again.'),
      },
    });
  } catch (error) {
    clearTimeout(timer);
    console.error(error);
    report(error instanceof Error ? error.message : 'Connection failed. Disconnect and reconnect to retry.');
  }
});

// A successful signaling event alone does not prove that media is arriving.
video.addEventListener('playing', () => {
  clearTimeout(timer);
  report('Connected');
  video.focus();
});

disconnect.addEventListener('click', () => {
  clearTimeout(timer);
  // Reload resets the SDK singleton and fetches any changed provider endpoint.
  AppStreamer.stop().catch(() => {});
  window.location.reload();
});

document.querySelector('#fullscreen').addEventListener('click', () => {
  const action = document.fullscreenElement
    ? document.exitFullscreen()
    : document.querySelector('#stream-container').requestFullscreen();
  action.catch(() => report('Fullscreen is unavailable in this browser.'));
});

// Keep browser refresh and developer tools usable while the stream has focus.
document.addEventListener('keydown', (event) => {
  if (event.key === 'F5' || ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'r')) {
    event.preventDefault();
    event.stopImmediatePropagation();
    window.location.reload();
  } else if (event.key === 'F12'
      || ((event.ctrlKey || event.metaKey) && event.shiftKey && ['i', 'j', 'c'].includes(event.key.toLowerCase()))
      || (event.metaKey && event.altKey && ['i', 'j', 'c'].includes(event.key.toLowerCase()))) {
    event.stopImmediatePropagation();
  }
}, true);

window.addEventListener('pagehide', () => {
  if (requested) AppStreamer.stop().catch(() => {});
});
