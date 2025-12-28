import LoadingSpinner from "@/components/custom/loading-spinner";

export default function LoadingScreen() {
  return (
    <div className="w-screen h-screen flex justify-center items-center">
      <LoadingSpinner />
    </div>
  );
}
