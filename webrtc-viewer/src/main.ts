import { AppStreamer, StreamType, eStatus, type StreamEvent } from '@nvidia/omniverse-webrtc-streaming-library';
import { streamConnection } from './connection.mjs';
import './style.css';

const status = document.querySelector<HTMLParagraphElement>('#status')!;
const connect = document.querySelector<HTMLButtonElement>('#connect')!;
const disconnect = document.querySelector<HTMLButtonElement>('#disconnect')!;
const video = document.querySelector<HTMLVideoElement>('#remote-video')!;
let requested = false;
let timer: ReturnType<typeof setTimeout> | undefined;

function report(message: string) {
  status.textContent = message;
}

function reportEnded(message: string) {
  clearTimeout(timer);
  report(message);
}

function onEvent(event: StreamEvent) {
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
    const response = await fetch('/connection.json', { cache: 'no-store' });
    if (!response.ok) throw new Error('Connection settings are unavailable. Restart the viewer command.');
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
  void AppStreamer.stop().catch(() => {});
  window.location.reload();
});

document.querySelector('#fullscreen')!.addEventListener('click', () => {
  const action = document.fullscreenElement
    ? document.exitFullscreen()
    : document.querySelector<HTMLElement>('#stream-container')!.requestFullscreen();
  void action.catch(() => report('Fullscreen is unavailable in this browser.'));
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
  if (requested) void AppStreamer.stop().catch(() => {});
});
