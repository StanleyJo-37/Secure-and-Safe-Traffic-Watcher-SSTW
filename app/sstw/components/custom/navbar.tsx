"use client";

import clsx from "clsx";
import Link from "next/link";
import { usePathname } from "next/navigation";

export interface NavItem {
  label: string;
  href: string;
  emphasis?: boolean;
}

export const navItems: Array<NavItem> = [
  {
    label: "Home",
    href: "/",
  },
  {
    label: "Motivations",
    href: "#motivations",
  },
  {
    label: "Features",
    href: "#features",
  },
  {
    label: "Technology",
    href: "#technology",
  },
  {
    label: "Demo",
    href: "/demo",
    emphasis: true,
  },
];

export default function Navbar() {
  const pathname = usePathname();

  if (pathname === "/demo") return <></>;

  return (
    <span className="flex flex-row items-center space-x-12 p-12 fixed left-auto right-16 z-50">
      {navItems.map((navItem, idx) => (
        <Link
          key={idx}
          className={clsx(
            "text-white text-xl px-3 py-1 duration-150 hover:underline",
            !navItem.emphasis && "hover:text-white/60",
            navItem.emphasis &&
              "bg-amber-500 rounded-md hover:bg-amber-700 duration"
          )}
          href={navItem.href}
        >
          {navItem.label}
        </Link>
      ))}
    </span>
  );
}
