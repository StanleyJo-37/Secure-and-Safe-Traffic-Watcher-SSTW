import { useState, VideoHTMLAttributes } from "react";
import { Dialog, DialogClose, DialogContent, DialogTitle } from "../ui/dialog";
import { Separator } from "../ui/separator";
import { Fullscreen } from "lucide-react";
import LoadingSpinner from "./loading-spinner";
import { cn } from "@/lib/utils";
import { Skeleton } from "../ui/skeleton";

export default function VideoPlayer({
  props,
  title = "Video",
}: {
  props: VideoHTMLAttributes<HTMLVideoElement>;
  title: string;
}) {
  const [isDialogOpen, setIsDialogOpen] = useState<boolean>(false);

  function VideoElementLoading() {
    return (
      <Skeleton className="w-md h-72 flex justify-center items-center">
        <LoadingSpinner />
      </Skeleton>
    );
  }

  function FullScreenDialog() {
    return (
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-none w-screen h-screen p-12 m-0 border-none bg-black flex flex-col">
          <DialogTitle>{title}</DialogTitle>
          <Separator />
          <video {...props} controls className="w-full h-full object-contain" />
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <div className="flex flex-row w-fit items-center space-x-4 relative">
      <FullScreenDialog />
      <div>
        {props.src ? (
          <video {...props} controls className="max-w-md" />
        ) : (
          <VideoElementLoading />
        )}
        <Fullscreen
          className="absolute right-4 top-4 text-white hover:text-white/25"
          onClick={() => {
            if (props.src) setIsDialogOpen(true);
          }}
        />
      </div>
    </div>
  );
}
