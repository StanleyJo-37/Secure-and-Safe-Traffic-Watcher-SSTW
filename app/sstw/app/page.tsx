import Image from "next/image";
import HeroCover from "@/assets/hero.jpg";
import {
  BrainCircuit,
  ChevronDown,
  Cpu,
  ShieldAlert,
  TrendingUp,
} from "lucide-react";
import {
  AnimatedTestimonials,
  Testimonial,
} from "@/components/ui/animated-testimonials";

const features: Array<Testimonial> = [
  {
    quote:
      "SSTW tracks and logs vehicles along with other information which creates a strutured dataset useful for trend analysis.",
    name: "Traffic Monitoring",
    src: "/monitor.jpg",
  },
  {
    quote:
      "SSTW tracks vehicles and can be used in an intersection. With learned features over time, SSTW can control traffic light for a better management.",
    name: "Traffic Light Control System",
    src: "/traffic-light.jpg",
  },
  {
    quote:
      "SSTW can be accompanied with another anomaly detection model which can detect anomalies, such as accidents, crashes, or anomolous behaviours using recorded data.",
    name: "Anomaly Detections",
    src: "/crash.jpg",
  },
];

export default function Page() {
  return (
    <div>
      <section className="relative h-screen flex flex-col items-start justify-center">
        <Image
          src={HeroCover}
          alt="Hero"
          fill
          className="object-cover -z-50"
          priority
        />

        <div className="absolute inset-0 -z-40 bg-linear-to-r from-black/80 to-transparent" />
        <div className="absolute inset-0 -z-40 bg-linear-to-t from-background/20 to-transparent" />

        <h1 className="text-white font-semibold text-8xl w-1/2 ml-16">
          Secure and Safe Traffic Watcher
        </h1>

        <div className="absolute bottom-8 left-1/2 -translate-x-1/2">
          <ChevronDown className="animate-bounce text-white w-10 h-10" />
        </div>
      </section>

      <section
        className="min-h-screen flex flex-col items-center justify-center"
        id="motivations"
      >
        <div className="text-center mb-16 max-w-2xl">
          <h2 className="text-orange-500 font-bold tracking-widest uppercase text-sm mb-4">
            Why We Built This
          </h2>
          <h3 className="text-5xl md:text-6xl font-semibold text-white mb-6 leading-tight">
            Traffic management <br />
            <span className="text-gray-500">needs an upgrade.</span>
          </h3>
          <p className="text-gray-400 text-lg">
            Traditional monitoring is reactive. We are building a system that is
            proactive, intelligent, and designed to save lives.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-6xl">
          <div className="group p-8 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all duration-300">
            <div className="h-12 w-12 bg-orange-500/20 rounded-lg flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <ShieldAlert className="text-orange-500 w-6 h-6" />
            </div>
            <h4 className="text-xl font-bold text-white mb-3">
              Enhance Safety
            </h4>
            <p className="text-gray-400 leading-relaxed">
              Reducing accident response times by detecting collisions the
              moment they happen, ensuring help arrives faster.
            </p>
          </div>

          <div className="group p-8 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all duration-300">
            <div className="h-12 w-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <TrendingUp className="text-blue-500 w-6 h-6" />
            </div>
            <h4 className="text-xl font-bold text-white mb-3">Optimize Flow</h4>
            <p className="text-gray-400 leading-relaxed">
              Analyzing traffic density in real-time to adjust signals and
              prevent congestion before it creates gridlock.
            </p>
          </div>

          <div className="group p-8 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all duration-300">
            <div className="h-12 w-12 bg-purple-500/20 rounded-lg flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <BrainCircuit className="text-purple-500 w-6 h-6" />
            </div>
            <h4 className="text-xl font-bold text-white mb-3">AI Precision</h4>
            <p className="text-gray-400 leading-relaxed">
              Replacing error-prone manual monitoring with computer vision that
              operates 24/7 without fatigue.
            </p>
          </div>
        </div>
      </section>

      <section
        className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden bg-zinc-900 py-24"
        id="features"
      >
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-125 bg-orange-500/10 blur-[120px] rounded-full pointer-events-none" />

        <div className="z-10 w-full max-w-7xl px-4 flex flex-col items-center">
          <h2 className="text-orange-500 font-bold tracking-widest uppercase text-sm mb-4">
            Capabilities
          </h2>
          <h3 className="text-4xl md:text-5xl font-semibold text-white mb-16 text-center">
            Powerful <span className="text-gray-500">Features</span>
          </h3>

          <div className="w-full">
            <AnimatedTestimonials testimonials={features} />
          </div>
        </div>
      </section>

      <section
        className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden bg-black py-24 px-6"
        id="technology"
      >
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-size-[24px_24px] pointer-events-none" />

        <div className="relative z-10 max-w-4xl w-full">
          <div className="text-center mb-12">
            <h2 className="text-orange-500 font-bold tracking-widest uppercase text-sm mb-4">
              Under the Hood
            </h2>
            <h3 className="text-4xl md:text-5xl font-semibold text-white">
              Technology <span className="text-gray-500">Stack</span>
            </h3>
          </div>

          <div className="relative group">
            <div className="absolute -inset-1 bg-linear-to-r from-orange-500/20 to-purple-500/20 rounded-3xl blur opacity-75 group-hover:opacity-100 transition duration-1000 group-hover:duration-200" />
            <div className="relative bg-zinc-900/80 backdrop-blur-xl ring-1 ring-white/10 rounded-3xl p-8 md:p-12">
              <div className="flex flex-col md:flex-row gap-8 items-center">
                <div className="shrink-0 p-6 bg-black/40 rounded-2xl border border-white/5">
                  <Cpu className="w-12 h-12 text-orange-500" />
                </div>

                <p className="text-lg md:text-xl text-gray-300 leading-relaxed font-light">
                  SSTW uses classical and State-Of-The-Art model to detect
                  vehicles. We are currently employing Selective Search + Linear
                  SVC and YOLO v8n for the detection model while utilizing
                  DeepSORT to keep track of vehicles passing by and/or waiting.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
