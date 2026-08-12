/**
 * Fill the explainer's numbers from the measured file.
 *
 * Every `data-fact` in the page ships with a plausible value already written
 * in, so the prose reads correctly with the network off or over `file://`.
 * This replaces those with what `scripts/site_facts.py` last measured, and
 * flags any that no longer match, because a number quoted from memory and a
 * number quoted from a run look identical to a reader and should not.
 */

export async function loadFacts(url = "data/facts.json") {
  let facts;
  try {
    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) throw new Error(`${response.status}`);
    facts = await response.json();
  } catch {
    return null; // opened from disk, or the file has not been generated
  }

  const stale = [];
  for (const node of document.querySelectorAll("[data-fact]")) {
    const key = node.dataset.fact;
    const measured = facts[key];
    if (measured === undefined) continue;
    const written = node.innerHTML.trim();
    if (written !== measured) stale.push(`${key}: page says "${written}", measured ${measured}`);
    node.innerHTML = measured;
  }

  if (stale.length) {
    console.info(
      `${stale.length} figure(s) in the page were behind the measured values and have been ` +
        `replaced:\n  ${stale.join("\n  ")}`
    );
  }
  return facts;
}
