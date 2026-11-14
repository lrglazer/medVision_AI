"use client";

import Link from "next/link";
import { Activity, Bone, Info, Menu, Workflow } from "lucide-react";
import { useState } from "react";

const links = [
  { href: "/", label: "Home", icon: Activity },
  { href: "/chest", label: "Chest X-ray Analysis", icon: Activity },
  { href: "/bone", label: "Bone X-ray Analysis", icon: Bone },
  { href: "/how-it-works", label: "How It Works", icon: Workflow },
  { href: "/about", label: "About", icon: Info },
];

export default function SiteHeader() {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#05070d]/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-2xl bg-gradient-to-br from-sky-400 to-violet-500 text-white shadow-sm">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm font-bold tracking-[0.16em] text-white">
              MEDVISION
            </p>
            <p className="text-xs text-slate-400">
              Explainable AI for medical imaging
            </p>
          </div>
        </Link>

        <nav className="hidden items-center gap-1 lg:flex">
          {links.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className="inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium text-slate-300 transition hover:bg-white/10 hover:text-white"
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          ))}
        </nav>

        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          className="grid h-10 w-10 place-items-center rounded-xl border border-white/10 lg:hidden"
          aria-label="Toggle navigation"
        >
          <Menu className="h-5 w-5" />
        </button>
      </div>

      {open && (
        <nav className="border-t border-white/10 bg-white/[0.03] px-4 py-3 lg:hidden">
          <div className="mx-auto flex max-w-7xl flex-col gap-1">
            {links.map(({ href, label, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                onClick={() => setOpen(false)}
                className="inline-flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium text-slate-300 hover:bg-white/10"
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
}
