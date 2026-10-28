---
name: deslop-design
description: Review and fix the visual and interaction design of demos, dashboards, Spaces and static pages so they look deliberately designed for the task, not like generic AI-generated UI, and verify the result with screenshots on desktop, mobile and dark mode.
---

# Deslop design (demos, Spaces, dashboards)

Use this when building or reviewing any page or app people will use: Gradio/HF Spaces, static HTML demos,
dashboards, report pages. Pair it with `deslop-writing` for the copy.

## Workflow

1. **Screenshot the current state** at desktop (1280 px) and mobile (390 px) width, light and dark mode if the page
   supports both. Use the real deployed page when possible (Playwright over Chrome CDP `http://localhost:29229`, or a
   local run of the app).
2. **Write down the page's one job** (for example: "pick a state, see which action the model picks and whether it
   was right"). Everything on screen either serves that job or is cut, collapsed, or moved below it.
3. **Audit** against the slop list and the rules below. Note each issue with its location.
4. **Fix**, re-screenshot at the same sizes, and compare before/after. Check the browser console for errors and
   that every control still works.

## Slop list (remove on sight)
- Gradient hero banners, glassmorphism, glow, drop-shadow stacks, oversized rounded "cards" around everything.
- Emoji or decorative icons in headings, buttons and tabs.
- Generic purple/indigo-on-white or neon-on-black palettes chosen by default; more than one accent colour.
- Walls of intro text above the interactive part; the user should reach the main control without scrolling.
- Badge/pill soup, "NEW"/"Beta" stickers, decorative stats counters.
- Center-aligned paragraphs and tables; mixed alignments.
- Placeholder or lorem copy, "Click the button below to get started!".
- Charts with default titles, unlabeled axes, no units, rainbow colour maps, legends covering data,
  3D or pie charts for comparisons.
- Tables with every column at 3+ decimals, unaligned numbers, internal variable names as headers.
- Spinners with no text for slow operations; silent failures.

## Rules
- **Hierarchy:** one H1 that names the thing in plain words, one sentence saying what it does, then the main
  interaction. Methods, caveats and licences go below or in a collapsible section, never deleted.
- **Typography:** system font stack, 15 to 16 px body, max line length about 75 characters, at most three text sizes.
- **Colour:** neutral background and text, one accent used only for the primary action and the key data series.
  Status colours (good/bad) must also differ in shape or label for colour-blind readers. Check WCAG AA contrast,
  including muted text and dark mode.
- **Layout:** left-aligned text, consistent spacing scale (4/8/16/24 px), content max-width about 1100 px,
  works at 390 px without horizontal scrolling (tables may scroll inside their container).
- **Controls:** label every control with what it changes and its unit/range. Sensible defaults so the first click
  shows a meaningful result. The primary button is the only filled button.
- **Data display:** right-align numbers, same precision per column, units in column headers, highlight the row or
  point the user should look at (for example the model's pick and the best option). Axis labels say the quantity
  and unit and which direction is better.
- **Feedback:** say how long slow calls take ("about 5 to 15 s on a shared GPU"), show progress text, and give a
  clear message plus a fallback when something fails (quota, timeout).
- **Honesty:** the UI must make clear what is live vs precomputed, simulated vs real, and whose model it is.

## Checks before delivering
- Does the first screen show the main control and explain the page in under 30 words?
- Any element from the slop list left?
- Do desktop, mobile and dark-mode screenshots all look intentional, with no overflow or low-contrast text?
- Does every chart axis have a label with units? Does every table column have units and consistent precision?
- Did the copy pass `deslop-writing`?
