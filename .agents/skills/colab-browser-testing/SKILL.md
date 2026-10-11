---
name: colab-browser-testing
description: Use when testing or executing Google Colab notebooks autonomously using a browser agent - handles connecting to T4 GPU runtimes, bypassing warning dialogs, mounting Google Drive, injecting code via Monaco editor, monitoring cell execution, and verifying outputs end-to-end
---

# Colab Browser Testing & Execution Guide

This skill teaches agents how to autonomously control, execute, and verify Google Colab notebooks using a browser agent (via Chrome DevTools MCP or browser subagents).

---

## When to Use

* When neural models (TTS, whole-body lip-sync, video diffusion, face restoration) require zero-cost cloud GPU compute (Tesla T4 16GB) on Google Colab.
* When testing that a committed Colab notebook (`.ipynb`) runs cleanly from scratch without human intervention.
* When executing end-to-end production runs, generating deliverables, and extracting verified outputs or screenshots back to the user.
* When debugging Colab runtime issues, out-of-memory (OOM) errors, or dependency conflicts in real-time.

---

## Architecture & Tool Requirements

Google Colab runs as a complex Single Page Application (SPA) with Monaco Editor instances and Shadow DOM components. Raw click/type operations often fail. The reliable approach uses direct JavaScript evaluation via `mcp_chrome_devtools_evaluate_script` or `browser_evaluate`.

Required MCP tools:
* `mcp_chrome_devtools_list_pages` / `mcp_chrome_devtools_new_page`
* `mcp_chrome_devtools_evaluate_script`
* `mcp_chrome_devtools_click`
* `mcp_chrome_devtools_take_snapshot`
* `mcp_chrome_devtools_take_screenshot`

---

## The End-to-End Execution Protocol

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Navigation & Page Discovery                              │
│ - Check active pages; open target notebook URL              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Runtime Connection & T4 GPU Verification                 │
│ - Bypass 'Warning: This notebook was not authored by Google'│
│ - Check/Switch runtime type to T4 GPU                        │
│ - Click Connect and probe !nvidia-smi                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Monaco Editor Injection & Execution                      │
│ - Inject code via window.monaco.editor.getModels()          │
│ - Pierce Shadow DOM to trigger colab-run-button             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Asynchronous State Monitoring & Output Harvesting        │
│ - Poll cell.classList.contains('running')                   │
│ - Extract text from output area and nested iframes          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Objective Verification & Screenshot Proof                 │
│ - Inspect JSON audit scorecards                             │
│ - Take screenshot of rendered video players / widgets       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Quota Conservation & Teardown                            │
│ - Unload models / flush VRAM                                │
│ - Explicitly disconnect runtime when batch finishes         │
└──────────────────────────────┬──────────────────────────────┘
```

---

## Step-by-Step Implementation

### Step 1: Navigating to the Target Notebook

Always check existing pages first to avoid spawning duplicate tabs:

```javascript
// Step 1: Check if notebook already open
const pages = await list_pages();
// If present, switch or reuse. Otherwise, create new page with URL:
// https://colab.research.google.com/github/<owner>/<repo>/blob/<branch>/quick_demo.ipynb
```

### Step 2: Bypassing Author Warning Dialogs

When opening GitHub notebooks, Colab frequently shows a warning dialog:
*"Warning: This notebook was not authored by Google."*

To bypass this automatically via JavaScript:
```javascript
() => {
  const buttons = Array.from(document.querySelectorAll('mwc-button, button'));
  const runAnyway = buttons.find(b => b.innerText.includes('Run anyway') || b.id === 'ok');
  if (runAnyway) {
    runAnyway.click();
    return 'Clicked Run anyway dialog';
  }
  return 'No warning dialog present';
}
```

### Step 3: Connecting to the T4 GPU Runtime

1. Check if the runtime is connected:
```javascript
() => {
  const statusBtn = document.querySelector('colab-connect-button');
  if (!statusBtn) return 'no connect button';
  const shadow = statusBtn.shadowRoot;
  const connectBtn = shadow?.querySelector('#connect') || shadow?.querySelector('button');
  const text = statusBtn.innerText || '';
  if (text.includes('RAM') && text.includes('Disk')) {
    return 'Already connected';
  }
  if (connectBtn) {
    connectBtn.click();
    return 'Clicked connect';
  }
  return 'Connect button status: ' + text;
}
```

2. Confirm T4 GPU accelerator is active:
Run a quick GPU probe cell:
```python
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```
Verify that output contains `Tesla T4` and `15360MiB`.

### Step 4: Injecting Code into Monaco Editor Cells

**CRITICAL RULE:** Do NOT use simulated keyboard typing on Colab code cells. Colab uses Monaco Editor, and keyboard emulation drops characters or triggers autocompletions.

Instead, locate the Monaco model and call `setValue`:

```javascript
() => {
  // Find all Monaco editor models in memory
  const models = window.monaco.editor.getModels();
  
  // Target code to inject
  const code = `
import sys
if '/content/app' not in sys.path:
    sys.path.insert(0, '/content/app')

from src.pipeline import VideoProductionPipeline
pipeline = VideoProductionPipeline(use_drive=True)
print("Pipeline ready!")
`;

  // Set code into the target cell's model
  // Note: cell model indices typically match the order of code cells
  models[targetModelIndex].setValue(code);

  // Trigger execution by clicking the cell's run button via Shadow DOM
  const cell = document.querySelectorAll('.cell, colab-cell')[targetCellIndex];
  const runBtn = cell.querySelector('colab-run-button');
  if (runBtn && runBtn.shadowRoot) {
    runBtn.shadowRoot.querySelector('#run-button').click();
    return 'Cell execution initiated';
  }
  return 'Run button not found';
}
```

### Step 5: Asynchronous Polling & Output Extraction

Colab executes cells asynchronously. Check the running state and harvest output from both standard DOM elements and sandboxed iframes:

```javascript
() => {
  const cell = document.querySelectorAll('.cell, colab-cell')[targetCellIndex];
  if (!cell) return { isRunning: false, error: 'Cell not found' };

  // Check running status
  const isRunning = cell.classList.contains('running') || 
                    cell.getAttribute('status') === 'running';

  // Extract direct text output
  let text = cell.querySelector('.output, colab-output-area, .output-area')?.innerText || '';

  // Extract iframe output (for Rich HTML, images, and video players)
  const iframe = cell.querySelector('iframe');
  let iframeText = '';
  if (iframe) {
    try {
      iframeText = iframe.contentDocument?.body?.innerText || '';
    } catch (e) {
      // Cross-origin iframe fallback
      iframeText = 'iframe present (sandboxed)';
    }
  }

  return {
    isRunning,
    text,
    iframeText,
    hasError: text.includes('Traceback') || text.includes('Error:')
  };
}
```

**Polling Loop Pattern:**
* Poll every 5 to 10 seconds.
* For long diffusion steps (LatentSync, GFPGAN), set a ceiling timeout of 8 to 10 minutes.
* Stop immediately if `hasError` is true.

### Step 6: Verifying Outputs & Capturing Visual Proof

Once the cell finishes running:
1. **JSON Audit Scorecard:** Extract the structured verification report (`_quality_audit.json`) containing audio SNR, face detection rate, and AV duration delta.
2. **Visual Snapshot:** Take a screenshot of the inline video player or scorecard:
   ```javascript
   // Use mcp_chrome_devtools_take_screenshot
   ```
3. **Google Drive Check:** Verify that the output exists in persistent storage:
   ```python
   !ls -lh /content/drive/MyDrive/video_gen_platform/outputs/
   ```

### Step 7: Teardown & GPU Quota Conservation

To prevent burning the user's rolling daily 3.5–5 hour T4 GPU quota:
1. Always unload models between stages (`pipeline.voice_cloner.unload()`, `torch.cuda.empty_cache()`).
2. When testing completes, unassign the runtime if no further jobs are queued:
   ```python
   from google.colab import runtime
   runtime.unassign()
   ```

---

## Anti-Patterns & Common Traps

| Anti-Pattern | Why It Breaks | Correct Pattern |
| :--- | :--- | :--- |
| **Simulated Keystrokes** | Monaco editor drops characters or corrupts indentations. | Use `monaco.editor.getModels()[i].setValue(code)`. |
| **Direct button click on `colab-run-button`** | Button is nested inside a Shadow Root (`#shadow-root`). | Pierce the shadow root: `runBtn.shadowRoot.querySelector('#run-button').click()`. |
| **Ignoring iframe outputs** | IPython HTML/widgets/video render inside sandboxed iframes. Direct `.innerText` misses them. | Inspect both `.output` and `cell.querySelector('iframe')`. |
| **Assuming Drive mounts silently** | First-time mount requires an OAuth approval popup. | Use cached mount tokens or handle the auth prompt if unassigned. |
| **Leaving Colab GPU idle** | Burns daily rolling quota without generating output. | Unassign runtime immediately after production finishes. |

---

## Verified Reusable Helper (Browser Agent Injector)

Use this complete JavaScript payload to safely inject and execute arbitrary code in any Colab notebook:

```javascript
(async (cellIdx, pythonCode) => {
  const cells = document.querySelectorAll('.cell, colab-cell');
  const targetCell = cells[cellIdx];
  if (!targetCell) throw new Error(`Cell ${cellIdx} not found`);

  // 1. Get or find Monaco model
  const models = window.monaco.editor.getModels();
  const codeCells = Array.from(cells).filter(c => c.querySelector('colab-run-button'));
  const modelIdx = codeCells.indexOf(targetCell);
  
  if (modelIdx >= 0 && modelIdx < models.length) {
    models[modelIdx].setValue(pythonCode);
  }

  // 2. Click Run Button
  const runBtn = targetCell.querySelector('colab-run-button');
  if (runBtn && runBtn.shadowRoot) {
    const innerBtn = runBtn.shadowRoot.querySelector('#run-button');
    innerBtn.click();
    return { status: 'triggered', cellIndex: cellIdx };
  }
  throw new Error('Run button shadow root not accessible');
})(targetCellIndex, pythonCode);
```
