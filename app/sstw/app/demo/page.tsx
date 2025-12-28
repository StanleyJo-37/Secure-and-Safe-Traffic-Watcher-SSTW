"use client";

import { FileUpload } from "@/components/ui/file-upload";
import { Button } from "@/components/ui/button";
import axios from "@/lib/axios";
import { useCallback, useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import LoadingSpinner from "@/components/custom/loading-spinner";
import { toast } from "sonner";
import { AxiosError } from "axios";
import { stringifyError } from "next/dist/shared/lib/utils";
import { cn } from "@/lib/utils";
import { DemoState, VidResult } from "@/app/types/demo-types";
import ResultCard from "@/components/custom/result-card";

let socket: any;

export default function Log() {
  const [videos, setVideos] = useState<Array<File>>([]);
  const [state, setState] = useState<DemoState>(DemoState.IDLE);
  const [videoResults, setVideoResults] = useState<Array<VidResult>>([]);
  const [openResult, setOpenResult] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    socket = io("http://localhost:5000", {
      transports: ["websocket"],
    });

    socket.on("task_completed", (args: any) => {
      setVideoResults((prev) =>
        prev.map((vid) =>
          vid.task_id === args.task_id
            ? {
                ...vid,
                video_url: args.video_url,
                tracked_vehicles: args.tracked_vehicles,
              }
            : vid
        )
      );
      setState(DemoState.SUCCESS);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const uploadVideosAndStartProcessing = useCallback(async () => {
    setOpenResult(true);
    if (videos.length === 0) return;

    try {
      setState(DemoState.IS_UPLOADING);

      const formData = new FormData();
      formData.append("video", videos[0]);

      const response = await axios.post("/tracker/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setVideoResults(response.data);

      setState(DemoState.IS_PROCESSING);
      setOpenResult(true);

      setTimeout(() => {
        document
          .getElementById("results-section")
          ?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } catch (err) {
      const message =
        err instanceof AxiosError
          ? err.message || err.response?.data.message
          : stringifyError(err as unknown as Error);
      toast.error(message);
      setState(DemoState.ERROR);
    }
  }, [videos]);

  return (
    <div className="m-0 p-0 flex flex-col">
      <section className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden bg-zinc-900 py-24">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-250 h-125 bg-orange-600/10 blur-[120px] rounded-full pointer-events-none" />
        <div className="absolute bottom-0 right-0 w-200 h-150 bg-blue-600/5 blur-[120px] rounded-full pointer-events-none" />

        <div className="z-10 w-full max-w-7xl px-4 flex flex-col items-center">
          <h2 className="text-orange-500 font-bold tracking-widest uppercase text-5xl mb-4">
            SSTW Demo
          </h2>
          <h3 className="text-lg font-semibold text-white mb-16 text-center">
            Upload <span className="text-gray-500">mp4 video</span> to try SSTW
            out!
          </h3>

          <div className="flex flex-col items-center justify-center space-y-8">
            <FileUpload
              key="videos-upload"
              accept="video/mp4"
              onChange={(files) => {
                setVideos(files);
                // setState(DemoState.IDLE);
              }}
              limit={1}
              fileInputRef={fileInputRef}
            />

            <Button
              className={cn(
                "w-full h-14 text-lg font-medium rounded-xl transition-all",
                videos.length > 0 &&
                  state !== DemoState.IS_PROCESSING &&
                  state !== DemoState.IS_UPLOADING
                  ? "bg-orange-500 hover:bg-orange-600 text-white shadow-lg shadow-orange-900/20"
                  : "bg-transparent text-zinc-500 border border-zinc-800 disabled:cursor-not-allowed disabled:opacity-50 disabled:pointer-events-auto"
              )}
              size="lg"
              disabled={
                videos.length === 0 ||
                state === DemoState.IS_PROCESSING ||
                state === DemoState.IS_UPLOADING
              }
              onClick={uploadVideosAndStartProcessing}
              variant={
                videos.length <= 0
                  ? "outline"
                  : state === DemoState.ERROR
                  ? "destructive"
                  : "default"
              }
            >
              {state === DemoState.IS_UPLOADING ? (
                <div className="flex items-center gap-2">
                  <LoadingSpinner /> <span>Uploading...</span>
                </div>
              ) : (
                "Start Processing"
              )}
            </Button>
          </div>
        </div>
      </section>
      {openResult && (
        <section className="min-h-screen flex flex-col items-center relative overflow-hidden bg-zinc-900 py-12">
          <h1 className="text-5xl text-white mb-12">Result</h1>
          <div className="flex flex-col gap-4 space-y-4 justify-start w-full">
            {videoResults.map((vid, key) => (
              <ResultCard key={key} vid={vid} state={state} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
