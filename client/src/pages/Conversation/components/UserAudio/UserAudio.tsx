import { FC, useCallback, useEffect, useRef, useState } from "react";
import { useSocketContext } from "../../SocketContext";
import { useUserAudio } from "../../hooks/useUserAudio";
import { ClientVisualizer } from "../AudioVisualizer/ClientVisualizer";
import { type ThemeType } from "../../hooks/useSystemTheme";
import { decodeMessage } from "../../../../protocol/encoder";

type UserAudioProps = {
  theme: ThemeType;
};
export const UserAudio: FC<UserAudioProps> = ({theme}) => {
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const { sendMessage, socketStatus, socket } = useSocketContext();
  const containerRef = useRef<HTMLDivElement>(null);
  // Muted while Odin is speaking — prevents mic bleed feeding back into the model
  const isMutedRef = useRef(false);

  // Listen for speaking/listening control messages from the server
  useEffect(() => {
    if (!socket) return;
    const handleMessage = (e: MessageEvent) => {
      const data = new Uint8Array(e.data);
      try {
        const msg = decodeMessage(data);
        if (msg.type === 'control') {
          if (msg.action === 'speaking') {
            isMutedRef.current = true;
          } else if (msg.action === 'listening') {
            isMutedRef.current = false;
          }
        }
      } catch {
        // ignore decode errors
      }
    };
    socket.addEventListener('message', handleMessage);
    return () => socket.removeEventListener('message', handleMessage);
  }, [socket]);

  const onRecordingStart = useCallback(() => {
    console.log("Recording started");
  }, []);

  const onRecordingStop = useCallback(() => {
    console.log("Recording stopped");
  }, []);

  const onRecordingChunk = useCallback(
    (chunk: Uint8Array) => {
      if (socketStatus !== "connected" || isMutedRef.current) {
        return;
      }
      sendMessage({
        type: "audio",
        data: chunk,
      });
    },
    [sendMessage, socketStatus],
  );

  const { startRecordingUser, stopRecording } = useUserAudio({
    constraints: {
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
      video: false,
    },
    onDataChunk: onRecordingChunk,
    onRecordingStart,
    onRecordingStop,
  });

  useEffect(() => {
    let res: Awaited<ReturnType<typeof startRecordingUser>>;
    if (socketStatus === "connected") {
      startRecordingUser().then(result => {
        if (result) {
          res = result;
          setAnalyser(result.analyser);
        }
      });
    }
    return () => {
      console.log("Stop recording called from somewhere else.");
      stopRecording();
      res?.source?.disconnect();
    };
  }, [startRecordingUser, stopRecording, socketStatus]);

  return (
    <div className="user-audio h-5/6 aspect-square" ref={containerRef}>
      <ClientVisualizer theme={theme} analyser={analyser} parent={containerRef}/>
    </div>
  );
};
