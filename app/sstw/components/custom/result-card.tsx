"use client";

import { DemoState, VidResult } from "@/app/types/demo-types";
import VideoPlayer from "./video-player";
import {
  CheckCircle,
  FileVideo,
  Hash,
  LucideMessageSquareWarning,
} from "lucide-react";
import LoadingSpinner from "./loading-spinner";

export default function ResultCard({
  vid,
  state,
}: {
  vid: VidResult;
  state: DemoState;
}) {
  return (
    <div className="flex flex-row space-x-4 px-8">
      <VideoPlayer title={vid.filename} props={{ src: vid.video_url }} />

      <div className="flex flex-col space-y-8">
        <div className="flex flex-col space-y-4">
          <div className="flex items-center gap-2 text-zinc-400 text-sm font-medium uppercase tracking-wider">
            <FileVideo className="w-4 h-4 text-orange-500" />
            <span>File Information</span>
          </div>
          <div className="space-y-2">
            <h3 className="text-2xl font-bold text-white">{vid.filename}</h3>
            <div className="flex items-center gap-2 text-zinc-500 font-mono text-sm bg-zinc-950/50 w-fit px-3 py-1 rounded-md border border-zinc-800">
              <Hash className="w-3 h-3" />
              {vid.task_id}
            </div>
          </div>
        </div>

        <div className="flex flex-col space-y-2 text-lg">
          <p>Current Status:</p>
          {state === DemoState.IS_PROCESSING && (
            <div className="text-yellow-400 flex flex-row items-center space-x-2">
              <LoadingSpinner /> <p>Processing...</p>
            </div>
          )}
          {state === DemoState.ERROR && (
            <div className="text-red-600 flex flex-row items-center space-x-2">
              <LucideMessageSquareWarning /> <p>Error occured.</p>
            </div>
          )}
          {state === DemoState.SUCCESS && (
            <div className="text-green-500 flex flex-row items-center space-x-2">
              <CheckCircle /> <p>Done processing!</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
