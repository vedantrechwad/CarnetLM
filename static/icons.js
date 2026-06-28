/** Material Symbols icon helpers for CarnetLM */
function icon(name, size = 20) {
    return `<span class="material-symbols-outlined" style="font-size:${size}px">${name}</span>`;
}

function refreshIcons(root) {
    // Material Symbols are a web font — no JS initialization needed.
    // This function is kept for backward compatibility with existing calls.
}

function sourceIcon(type) {
    const map = {
        Website: 'language',
        YouTube: 'play_circle',
        Clipboard: 'content_paste',
        Document: 'description',
    };
    return map[type] || 'description';
}
