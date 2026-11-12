import { useMemo, useState, useRef } from "react";
import { Photo } from "./api";
import { Cell, SelectMods } from "./Library";

interface DateViewProps {
  photos: Photo[];
  selected: number[];
  active: Photo | null;
  workingIds?: number[];
  busyAiIds?: number[];
  size: number;
  onSelect: (id: number, mods: SelectMods) => void;
  onOpenDevelop: (id: number) => void;
  onOpenViewer: (id: number) => void;
  onSelectMultiple?: (ids: number[]) => void;
}

type GroupBy = "month" | "day" | "year";

interface DateGroup {
  key: string;
  label: string;
  sublabel: string;
  year: string;
  photos: Photo[];
}

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

const WEEKDAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

export function DateView({
  photos,
  selected,
  active,
  workingIds,
  busyAiIds,
  size,
  onSelect,
  onOpenDevelop,
  onOpenViewer,
  onSelectMultiple,
}: DateViewProps) {
  const [groupBy, setGroupBy] = useState<GroupBy>("month");
  const [selectedYear, setSelectedYear] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const containerRef = useRef<HTMLDivElement>(null);

  // Group photos chronologically
  const { groups, undated, years, totalDated } = useMemo(() => {
    const dated: Photo[] = [];
    const undatedList: Photo[] = [];

    for (const p of photos) {
      if (p.captured_at && p.captured_at.trim() && !p.captured_at.startsWith("0000")) {
        dated.push(p);
      } else {
        undatedList.push(p);
      }
    }

    // Sort descending by capture date
    dated.sort((a, b) => (b.captured_at || "").localeCompare(a.captured_at || ""));

    const yearSet = new Set<string>();
    const groupMap = new Map<string, DateGroup>();

    for (const p of dated) {
      const dateStr = p.captured_at!.replace("T", " ");
      const parts = dateStr.slice(0, 10).split(/[-:]/);
      const y = parts[0] || "Unknown";
      const m = parts[1] || "01";
      const d = parts[2] || "01";
      yearSet.add(y);

      let key = "";
      let label = "";
      let sublabel = "";

      if (groupBy === "year") {
        key = y;
        label = y;
        sublabel = `${y} Collection`;
      } else if (groupBy === "day") {
        key = `${y}-${m}-${d}`;
        const dt = new Date(parseInt(y, 10), parseInt(m, 10) - 1, parseInt(d, 10));
        const dayName = isNaN(dt.getTime()) ? "" : WEEKDAY_NAMES[dt.getDay()] || "";
        const monthName = MONTH_NAMES[parseInt(m, 10) - 1] || m;
        label = `${dayName ? `${dayName}, ` : ""}${monthName} ${parseInt(d, 10)}, ${y}`;
        sublabel = `${parts[0]}-${parts[1]}-${parts[2]}`;
      } else {
        // month (default)
        key = `${y}-${m}`;
        const monthName = MONTH_NAMES[parseInt(m, 10) - 1] || m;
        label = `${monthName} ${y}`;
        sublabel = `${y} · ${monthName}`;
      }

      if (!groupMap.has(key)) {
        groupMap.set(key, {
          key,
          label,
          sublabel,
          year: y,
          photos: [],
        });
      }
      groupMap.get(key)!.photos.push(p);
    }

    const sortedYears = Array.from(yearSet).sort().reverse();
    const allGroups = Array.from(groupMap.values());

    return {
      groups: allGroups,
      undated: undatedList,
      years: sortedYears,
      totalDated: dated.length,
    };
  }, [photos, groupBy]);

  const filteredGroups = useMemo(() => {
    if (!selectedYear) return groups;
    return groups.filter((g) => g.year === selectedYear);
  }, [groups, selectedYear]);

  const toggleCollapse = (key: string) => {
    setCollapsed((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSelectGroup = (groupPhotos: Photo[]) => {
    if (!onSelectMultiple) {
      if (groupPhotos.length > 0) {
        onSelect(groupPhotos[0].id, { shift: false, toggle: false });
      }
      return;
    }
    const ids = groupPhotos.map((p) => p.id);
    onSelectMultiple(ids);
  };

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-neutral-950">
      {/* Sub-toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 bg-neutral-900/90 px-4 py-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-neutral-200">📅 By Date</span>
          <span className="text-neutral-500">·</span>
          <span className="text-neutral-400">
            {totalDated} dated photo{totalDated === 1 ? "" : "s"} across {years.length} year{years.length === 1 ? "" : "s"}
          </span>
          {undated.length > 0 && (
            <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] text-neutral-400">
              +{undated.length} undated
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center rounded-md border border-neutral-700/60 bg-neutral-800 p-0.5">
            <button
              type="button"
              onClick={() => setGroupBy("year")}
              className={`rounded px-2 py-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
                groupBy === "year"
                  ? "bg-[var(--theme-accent,#0ea5e9)] font-semibold text-white shadow-xs"
                  : "text-neutral-400 hover:text-neutral-200"
              }`}
            >
              Years
            </button>
            <button
              type="button"
              onClick={() => setGroupBy("month")}
              className={`rounded px-2 py-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
                groupBy === "month"
                  ? "bg-[var(--theme-accent,#0ea5e9)] font-semibold text-white shadow-xs"
                  : "text-neutral-400 hover:text-neutral-200"
              }`}
            >
              Months
            </button>
            <button
              type="button"
              onClick={() => setGroupBy("day")}
              className={`rounded px-2 py-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
                groupBy === "day"
                  ? "bg-[var(--theme-accent,#0ea5e9)] font-semibold text-white shadow-xs"
                  : "text-neutral-400 hover:text-neutral-200"
              }`}
            >
              Days
            </button>
          </div>
        </div>
      </div>

      {/* Year Filter Pills */}
      {years.length > 0 && (
        <div className="flex items-center gap-1.5 overflow-x-auto border-b border-neutral-800/80 bg-neutral-900/40 px-4 py-1.5 text-[11px]">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500 mr-1">
            Jump to:
          </span>
          <button
            type="button"
            onClick={() => setSelectedYear(null)}
            className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
              selectedYear === null
                ? "bg-[var(--theme-accent,#0ea5e9)] text-white font-semibold"
                : "bg-neutral-800 text-neutral-400 hover:bg-neutral-700 hover:text-neutral-200"
            }`}
          >
            All Years ({totalDated})
          </button>
          {years.map((y) => {
            const count = groups.filter((g) => g.year === y).reduce((acc, g) => acc + g.photos.length, 0);
            return (
              <button
                key={y}
                type="button"
                onClick={() => {
                  setSelectedYear(selectedYear === y ? null : y);
                }}
                className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
                  selectedYear === y
                    ? "bg-[var(--theme-accent,#0ea5e9)] text-white font-semibold"
                    : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700 hover:text-white"
                }`}
              >
                {y} ({count})
              </button>
            );
          })}
        </div>
      )}

      {/* Timeline Scroll Area */}
      <div ref={containerRef} className="flex-1 overflow-y-auto p-4 space-y-6">
        {filteredGroups.length === 0 && undated.length === 0 ? (
          <div className="flex h-64 flex-col items-center justify-center gap-2 text-neutral-500">
            <span className="text-3xl">📅</span>
            <p className="text-sm font-medium text-neutral-300">No photos in this date range</p>
            <p className="text-xs text-neutral-500">Import photos with capture timestamps to browse chronologically.</p>
          </div>
        ) : null}

        {filteredGroups.map((group) => {
          const isCollapsed = !!collapsed[group.key];
          return (
            <section
              key={group.key}
              id={`date-group-${group.key}`}
              className="rounded-lg border border-neutral-800/80 bg-neutral-900/60 p-3 shadow-xs transition-all"
            >
              {/* Group Header */}
              <div className="sticky top-0 z-10 flex items-center justify-between rounded-md bg-neutral-900/95 px-2.5 py-1.5 backdrop-blur-md">
                <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => toggleCollapse(group.key)}>
                  <button
                    type="button"
                    className="flex h-5 w-5 items-center justify-center rounded text-neutral-400 hover:text-white transition-colors"
                    title={isCollapsed ? "Expand date group" : "Collapse date group"}
                  >
                    <span className="text-[10px] transform transition-transform">{isCollapsed ? "▶" : "▼"}</span>
                  </button>
                  <div>
                    <h3 className="text-xs font-semibold text-neutral-100 flex items-center gap-2">
                      <span>{group.label}</span>
                      <span className="rounded bg-neutral-800 px-1.5 py-0.2 text-[10px] font-normal text-neutral-400">
                        {group.photos.length} {group.photos.length === 1 ? "photo" : "photos"}
                      </span>
                    </h3>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => handleSelectGroup(group.photos)}
                    className="rounded bg-neutral-800 px-2 py-1 text-[10px] font-medium text-neutral-300 hover:bg-neutral-700 hover:text-white transition-colors cursor-pointer"
                    title="Select all photos in this date group"
                  >
                    Select All ({group.photos.length})
                  </button>
                </div>
              </div>

              {/* Photo Grid for this Date Group */}
              {!isCollapsed && (
                <div
                  className="mt-3 grid gap-2"
                  style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${size}px, 1fr))` }}
                >
                  {group.photos.map((ph) => (
                    <Cell
                      key={ph.id}
                      photo={ph}
                      size={size}
                      selected={selected.includes(ph.id)}
                      active={active?.id === ph.id}
                      inWorking={workingIds?.includes(ph.id)}
                      aiBusy={busyAiIds?.includes(ph.id)}
                      onClick={(m) => onSelect(ph.id, m)}
                      onOpen={() => onOpenDevelop(ph.id)}
                      onView={() => {
                        onSelect(ph.id, { shift: false, toggle: false });
                        onOpenViewer(ph.id);
                      }}
                    />
                  ))}
                </div>
              )}
            </section>
          );
        })}

        {/* Undated Section */}
        {(!selectedYear || selectedYear === "Undated") && undated.length > 0 && (
          <section
            id="date-group-undated"
            className="rounded-lg border border-neutral-800/80 bg-neutral-900/40 p-3 shadow-xs"
          >
            <div className="flex items-center justify-between rounded-md bg-neutral-900/90 px-2.5 py-1.5">
              <div className="flex items-center gap-2 cursor-pointer" onClick={() => toggleCollapse("undated")}>
                <span className="text-[10px] text-neutral-400">{collapsed["undated"] ? "▶" : "▼"}</span>
                <h3 className="text-xs font-semibold text-neutral-300">
                  Undated Photos
                  <span className="ml-2 rounded bg-neutral-800 px-1.5 py-0.2 text-[10px] font-normal text-neutral-400">
                    {undated.length} photo{undated.length === 1 ? "" : "s"}
                  </span>
                </h3>
              </div>
              <button
                type="button"
                onClick={() => handleSelectGroup(undated)}
                className="rounded bg-neutral-800 px-2 py-1 text-[10px] font-medium text-neutral-300 hover:bg-neutral-700 hover:text-white transition-colors cursor-pointer"
              >
                Select All ({undated.length})
              </button>
            </div>

            {!collapsed["undated"] && (
              <div
                className="mt-3 grid gap-2"
                style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${size}px, 1fr))` }}
              >
                {undated.map((ph) => (
                  <Cell
                    key={ph.id}
                    photo={ph}
                    size={size}
                    selected={selected.includes(ph.id)}
                    active={active?.id === ph.id}
                    inWorking={workingIds?.includes(ph.id)}
                    aiBusy={busyAiIds?.includes(ph.id)}
                    onClick={(m) => onSelect(ph.id, m)}
                    onOpen={() => onOpenDevelop(ph.id)}
                    onView={() => {
                      onSelect(ph.id, { shift: false, toggle: false });
                      onOpenViewer(ph.id);
                    }}
                  />
                ))}
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
