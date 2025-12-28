import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/custom/navbar";
import { Manrope } from "next/font/google";
import { Toaster } from "@/components/ui/sonner";

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope",
});

export const metadata: Metadata = {
  title: "SSTW - Traffic Watcher",
  description: "Secure and Safe Traffic Watcher - AI-powered traffic analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${manrope.variable} font-sans antialiased`}>
        <Navbar />
        {children}
        <Toaster />
      </body>
    </html>
  );
}
