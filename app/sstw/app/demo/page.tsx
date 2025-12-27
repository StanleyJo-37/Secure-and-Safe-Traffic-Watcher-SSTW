"use client";

import { FileUpload } from "@/components/ui/file-upload";
import { Button } from "@/components/ui/stateful-button";
import axios from "@/lib/axios";
import { useCallback, useEffect, useState } from "react";
import { io } from "socket.io-client";

enum State {
  IDLE,
  ERROR,
  IS_UPLOADING,
  IS_PROCESSING,
  IS_DOWNLOADING,
}

let socket: any;

export default function Log() {
  const [videos, setVideos] = useState<Array<File>>([]);
  const [state, setState] = useState<State>(State.IDLE);
  const [videoUrls, setVideoUrls] = useState<Array<string>>([]);

  useEffect(() => {
    socket = io({
      transports: ["websocket"],
    });

    socket = io("http://localhost:5000", {
      transports: ["websocket"],
    });

    socket.on("task_completed", (args: any) => {
      console.log("Video is ready at:", args.video_url);

      setVideoUrls((prev) => [...prev, args.video_url]);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const uploadVideosAndStartProcessing = useCallback(async () => {
    if (videos.length === 0) return;

    const resp = await axios.post("tracker/upload");

    try {
      setState(State.IS_UPLOADING);

      const formData = new FormData();
      videos.forEach((video) => {
        formData.append("videos", video);
      });

      const response = await axios.post("/tracker/upload", formData, {
        headers: { Accept: "multipart/form-data" },
      });

      // If upload succeeds, move to processing state
      setState(State.IS_PROCESSING);
    } catch (err) {
      setState(State.ERROR);
    }

    socket.on(
      "task_completed",
      async (args: { task_id: string; status: string; video_url: string }) => {
        setVideoUrls((prev) => [...prev, args.video_url]);
      }
    );
  }, []);

  return (
    <div className="m-0 p-0 flex flex-col">
      <section className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden bg-zinc-900 py-24">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-125 bg-orange-500/10 blur-[120px] rounded-full pointer-events-none" />

        <div className="z-10 w-full max-w-7xl px-4 flex flex-col items-center">
          <h2 className="text-orange-500 font-bold tracking-widest uppercase text-sm mb-4">
            Demo
          </h2>
          <h3 className="text-4xl md:text-5xl font-semibold text-white mb-16 text-center">
            Upload <span className="text-gray-500">mp4 video(s)</span> to try
            SSTW out!
          </h3>

          <FileUpload
            key="videos-upload"
            accept="video/mp4"
            onChange={(files) => setVideos(files)}
          />
        </div>

        {videos.length > 0 && (
          <Button
            disabled={
              state === State.IS_PROCESSING || state === State.IS_UPLOADING
            }
            onClick={uploadVideosAndStartProcessing}
          >
            Submit {videos.length} video{videos.length > 1 && "s"}
          </Button>
        )}
      </section>
      {state !== State.IDLE && state !== State.ERROR && (
        <section className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden bg-zinc-900 blur-[120px] py-24"></section>
      )}
    </div>
  );
}
