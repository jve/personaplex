import { useEffect, useRef } from 'react';

/**
 * Maintains a persistent SSE connection to /api/events.
 * When the server pushes a "wake" event (e.g. triggered by POST /api/notify),
 * onWake is called with the event text so the caller can start a session.
 * Reconnects automatically with a 5-second backoff on disconnect.
 */
export const useNotificationChannel = (
  serverAddr: string,
  onWake: (text: string) => void,
) => {
  // Keep onWake stable across re-renders without restarting the effect
  const onWakeRef = useRef(onWake);
  onWakeRef.current = onWake;

  useEffect(() => {
    const addr =
      serverAddr && serverAddr !== 'same' && serverAddr !== ''
        ? serverAddr
        : `${window.location.hostname}:${window.location.port}`;
    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    const url = `${protocol}//${addr}/api/events`;

    let es: EventSource | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    function connect() {
      if (cancelled) return;
      es = new EventSource(url);

      es.onmessage = (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          if (data.type === 'wake' && data.text) {
            onWakeRef.current(data.text);
          }
        } catch {
          // ignore malformed events
        }
      };

      es.onerror = () => {
        es?.close();
        es = null;
        if (!cancelled) {
          reconnectTimer = setTimeout(connect, 5000);
        }
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      es?.close();
      es = null;
    };
  }, [serverAddr]);
};
