import torch
import json
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_entropy(attention_weights):
    eps = 1e-10
    entropy = -np.sum(attention_weights * np.log(attention_weights + eps), axis=-1)
    return entropy


def visualize_interactive(model_name="Qwen/Qwen2.5-0.5B", text=None):
    print(f"Loading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Trying fallback model: gpt2")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)

    if text is None:
        text = "The quick brown fox jumps over the lazy dog."

    print(f"Processing text: '{text}'")
    inputs = tokenizer(text, return_tensors="pt")

    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    display_tokens = [t.replace('Ġ', ' ').replace('Ċ', '\\n') for t in tokens]

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    print(f"Model has {num_layers} layers and {num_heads} heads per layer")

    all_attention_data = []
    entropy_data = []
    
    for layer_idx in range(num_layers):
        layer_attention = attentions[layer_idx].squeeze(0).numpy().tolist()
        all_attention_data.append(layer_attention)
        
        layer_np = attentions[layer_idx].squeeze(0).numpy()
        layer_entropy = []
        for head_idx in range(num_heads):
            head_attention = layer_np[head_idx]
            head_entropy = compute_entropy(head_attention)
            avg_entropy = float(np.mean(head_entropy))
            layer_entropy.append(avg_entropy)
        entropy_data.append(layer_entropy)

    viz_data = {
        "tokens": display_tokens,
        "attention": all_attention_data,
        "entropy": entropy_data,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "model_name": model_name
    }

    print("Generating interactive visualization...")
    html_content = generate_html(viz_data)

    output_file = "attention_interactive.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Interactive visualization saved to {os.path.abspath(output_file)}")


def generate_html(data):
    data_json = json.dumps(data)
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attention Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 13px;
            color: #333;
            background: #fff;
            line-height: 1.4;
        }}
        .header {{
            padding: 12px 20px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #fafafa;
        }}
        .header h1 {{
            font-size: 14px;
            font-weight: 600;
            color: #333;
        }}
        .meta {{
            display: flex;
            gap: 20px;
            font-size: 12px;
            color: #666;
        }}
        .meta strong {{ color: #333; }}
        .container {{
            display: flex;
            height: calc(100vh - 45px);
        }}
        .controls {{
            width: 220px;
            padding: 16px;
            border-right: 1px solid #e0e0e0;
            background: #fafafa;
            overflow-y: auto;
        }}
        .control-group {{
            margin-bottom: 20px;
        }}
        .control-group label {{
            display: block;
            font-size: 11px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .layer-display {{
            display: flex;
            align-items: baseline;
            gap: 4px;
            margin-bottom: 8px;
        }}
        .layer-num {{
            font-size: 24px;
            font-weight: 600;
            color: #333;
        }}
        .layer-total {{
            font-size: 12px;
            color: #999;
        }}
        input[type="range"] {{
            width: 100%;
            height: 4px;
            -webkit-appearance: none;
            background: #ddd;
            border-radius: 2px;
            outline: none;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            background: #666;
            border-radius: 50%;
            cursor: pointer;
        }}
        input[type="range"]::-webkit-slider-thumb:hover {{
            background: #333;
        }}
        .btn-row {{
            display: flex;
            gap: 6px;
            margin-top: 10px;
        }}
        .btn {{
            flex: 1;
            padding: 6px 10px;
            font-size: 11px;
            font-weight: 500;
            border: 1px solid #ccc;
            background: #fff;
            color: #333;
            border-radius: 3px;
            cursor: pointer;
        }}
        .btn:hover {{
            background: #f5f5f5;
            border-color: #999;
        }}
        .btn.active {{
            background: #333;
            color: #fff;
            border-color: #333;
        }}
        select {{
            width: 100%;
            padding: 5px 8px;
            font-size: 12px;
            border: 1px solid #ccc;
            border-radius: 3px;
            background: #fff;
            margin-top: 8px;
        }}
        .head-grid {{
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 3px;
        }}
        .head-btn {{
            aspect-ratio: 1;
            font-size: 10px;
            font-weight: 500;
            border: 1px solid #ddd;
            background: #fff;
            color: #666;
            border-radius: 2px;
            cursor: pointer;
        }}
        .head-btn:hover {{
            border-color: #999;
            color: #333;
        }}
        .head-btn.active {{
            background: #333;
            color: #fff;
            border-color: #333;
        }}
        .chart-box {{
            height: 100px;
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 3px;
            padding: 8px;
        }}
        .main {{
            flex: 1;
            padding: 16px;
            overflow-y: auto;
            background: #fff;
        }}
        .section {{
            margin-bottom: 24px;
        }}
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .section-title {{
            font-size: 12px;
            font-weight: 600;
            color: #333;
        }}
        .section-subtitle {{
            font-size: 11px;
            color: #999;
        }}
        .matrix-container {{
            display: flex;
            gap: 24px;
        }}
        .matrix-wrapper {{
            flex: 1;
        }}
        .matrix-label {{
            font-size: 11px;
            font-weight: 500;
            color: #666;
            margin-bottom: 8px;
        }}
        .matrix {{
            display: inline-block;
        }}
        .matrix-header {{
            display: flex;
            margin-left: 50px;
            margin-bottom: 2px;
        }}
        .matrix-header span {{
            width: 36px;
            font-size: 9px;
            color: #999;
            text-align: center;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .matrix-row {{
            display: flex;
            align-items: center;
            height: 24px;
        }}
        .matrix-row-label {{
            width: 50px;
            font-size: 10px;
            color: #666;
            text-align: right;
            padding-right: 6px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .matrix-cell {{
            width: 36px;
            height: 22px;
            margin: 1px;
            border-radius: 2px;
            cursor: pointer;
            position: relative;
            transition: transform 0.1s;
        }}
        .matrix-cell:hover {{
            transform: scale(1.15);
            z-index: 10;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .matrix-cell.masked {{
            background: #f5f5f5 !important;
            cursor: default;
        }}
        .matrix-cell.masked:hover {{
            transform: none;
            box-shadow: none;
        }}
        .token-pills {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-bottom: 12px;
        }}
        .token-pill {{
            padding: 4px 10px;
            font-size: 11px;
            border: 1px solid #ddd;
            background: #fff;
            color: #666;
            border-radius: 12px;
            cursor: pointer;
        }}
        .token-pill:hover {{
            border-color: #999;
            color: #333;
        }}
        .token-pill.active {{
            background: #333;
            color: #fff;
            border-color: #333;
        }}
        .dist-chart {{
            height: 150px;
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 3px;
            padding: 12px;
        }}
        .tooltip {{
            position: fixed;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            z-index: 1000;
            pointer-events: none;
            min-width: 200px;
            opacity: 0;
            transition: opacity 0.15s;
        }}
        .tooltip.visible {{ opacity: 1; }}
        .tooltip-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #eee;
        }}
        .tooltip-pair {{
            font-size: 12px;
            font-weight: 500;
            color: #333;
        }}
        .tooltip-value {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        .tooltip-chart {{
            height: 60px;
            margin-bottom: 8px;
        }}
        .tooltip-list {{
            font-size: 11px;
        }}
        .tooltip-row {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 4px;
        }}
        .tooltip-rank {{
            width: 14px;
            height: 14px;
            background: #f0f0f0;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            font-weight: 600;
            color: #666;
        }}
        .tooltip-token {{
            flex: 1;
            color: #333;
        }}
        .tooltip-bar {{
            width: 50px;
            height: 4px;
            background: #eee;
            border-radius: 2px;
            overflow: hidden;
        }}
        .tooltip-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #a855f7);
        }}
        .tooltip-pct {{
            width: 35px;
            text-align: right;
            color: #999;
        }}
        .view-toggle {{
            display: flex;
            border: 1px solid #ddd;
            border-radius: 3px;
            overflow: hidden;
        }}
        .view-toggle button {{
            padding: 4px 12px;
            font-size: 11px;
            border: none;
            background: #fff;
            color: #666;
            cursor: pointer;
        }}
        .view-toggle button:not(:last-child) {{
            border-right: 1px solid #ddd;
        }}
        .view-toggle button.active {{
            background: #333;
            color: #fff;
        }}
    </style>
</head>
<body>
    <header class="header">
        <h1>Attention Visualization</h1>
        <div class="meta">
            <span>Model: <strong id="model-name"></strong></span>
            <span>Layers: <strong id="total-layers"></strong></span>
            <span>Heads: <strong id="total-heads"></strong></span>
            <span>Tokens: <strong id="token-count"></strong></span>
        </div>
    </header>
    
    <div class="container">
        <aside class="controls">
            <div class="control-group">
                <label>Layer</label>
                <div class="layer-display">
                    <span class="layer-num" id="current-layer">0</span>
                    <span class="layer-total">/ <span id="layer-max">0</span></span>
                </div>
                <input type="range" id="layer-slider" min="0" max="0" value="0">
                <div class="btn-row">
                    <button class="btn" id="play-btn">▶ Play</button>
                    <button class="btn" id="reset-btn">↺ Reset</button>
                </div>
                <select id="speed-select">
                    <option value="1500">Speed: Slow</option>
                    <option value="800" selected>Speed: Normal</option>
                    <option value="400">Speed: Fast</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Entropy (per head)</label>
                <div class="chart-box">
                    <canvas id="entropy-chart"></canvas>
                </div>
            </div>
            
            <div class="control-group">
                <label>Head</label>
                <div class="head-grid" id="head-grid"></div>
            </div>
            
            <div class="control-group">
                <label>View</label>
                <div class="view-toggle">
                    <button id="single-btn" class="active">Single</button>
                    <button id="compare-btn">Compare</button>
                </div>
            </div>
        </aside>
        
        <main class="main">
            <div class="section">
                <div class="section-header">
                    <span class="section-title">Attention Matrix</span>
                    <span class="section-subtitle" id="matrix-info">Layer 0</span>
                </div>
                <div class="matrix-container" id="matrix-container">
                    <div class="matrix-wrapper" id="primary-wrapper">
                        <div class="matrix-label" id="primary-label">Head 0</div>
                        <div class="matrix" id="primary-matrix"></div>
                    </div>
                    <div class="matrix-wrapper" id="compare-wrapper" style="display:none;">
                        <div class="matrix-label" id="compare-label">Head 1</div>
                        <div class="matrix" id="compare-matrix"></div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">
                    <span class="section-title">Attention Distribution</span>
                    <span class="section-subtitle">Select a source token to see where it attends</span>
                </div>
                <div class="token-pills" id="token-pills"></div>
                <div class="dist-chart">
                    <canvas id="dist-chart"></canvas>
                </div>
            </div>
        </main>
    </div>
    
    <div class="tooltip" id="tooltip">
        <div class="tooltip-header">
            <span class="tooltip-pair" id="tooltip-pair"></span>
            <span class="tooltip-value" id="tooltip-value"></span>
        </div>
        <div class="tooltip-chart">
            <canvas id="tooltip-chart"></canvas>
        </div>
        <div class="tooltip-list" id="tooltip-list"></div>
    </div>
    
    <script>
        const DATA = {data_json};
        
        let state = {{
            layer: 0,
            head: 0,
            compareHead: 1,
            token: 0,
            playing: false,
            interval: null,
            compare: false
        }};
        
        let entropyChart, distChart, tooltipChart;
        
        document.addEventListener('DOMContentLoaded', init);
        
        function init() {{
            document.getElementById('model-name').textContent = DATA.model_name;
            document.getElementById('total-layers').textContent = DATA.num_layers;
            document.getElementById('total-heads').textContent = DATA.num_heads;
            document.getElementById('token-count').textContent = DATA.tokens.length;
            document.getElementById('layer-max').textContent = DATA.num_layers - 1;
            
            const slider = document.getElementById('layer-slider');
            slider.max = DATA.num_layers - 1;
            slider.oninput = e => {{ state.layer = +e.target.value; update(); }};
            
            document.getElementById('play-btn').onclick = togglePlay;
            document.getElementById('reset-btn').onclick = () => {{ stopPlay(); state.layer = 0; update(); }};
            
            const grid = document.getElementById('head-grid');
            for (let i = 0; i < DATA.num_heads; i++) {{
                const btn = document.createElement('button');
                btn.className = 'head-btn' + (i === 0 ? ' active' : '');
                btn.textContent = i;
                btn.onclick = () => selectHead(i);
                grid.appendChild(btn);
            }}
            
            document.getElementById('single-btn').onclick = () => setCompare(false);
            document.getElementById('compare-btn').onclick = () => setCompare(true);
            
            const pills = document.getElementById('token-pills');
            DATA.tokens.forEach((t, i) => {{
                const pill = document.createElement('button');
                pill.className = 'token-pill' + (i === 0 ? ' active' : '');
                pill.textContent = t.trim() || '␣';
                pill.onclick = () => selectToken(i);
                pills.appendChild(pill);
            }});
            
            initCharts();
            update();
        }}
        
        function getEntropyColors(n) {{
            const colors = [];
            for (let i = 0; i < n; i++) {{
                const hue = (i / n) * 280 + 200;
                colors.push(`hsl(${{hue % 360}}, 65%, 55%)`);
            }}
            return colors;
        }}
        
        function getDistColors(values) {{
            return values.map(v => {{
                const hue = 220 + v * 60;
                const sat = 60 + v * 20;
                const light = 45 + (1 - v) * 20;
                return `hsl(${{hue}}, ${{sat}}%, ${{light}}%)`;
            }});
        }}
        
        function initCharts() {{
            const entropyColors = getEntropyColors(DATA.num_heads);
            
            entropyChart = new Chart(document.getElementById('entropy-chart'), {{
                type: 'bar',
                data: {{
                    labels: Array.from({{length: DATA.num_heads}}, (_, i) => i),
                    datasets: [{{ data: DATA.entropy[0], backgroundColor: entropyColors, borderRadius: 2 }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 8 }}, color: '#999' }} }},
                        y: {{ display: false }}
                    }}
                }}
            }});
            
            distChart = new Chart(document.getElementById('dist-chart'), {{
                type: 'bar',
                data: {{
                    labels: DATA.tokens.map(t => t.trim() || '␣'),
                    datasets: [{{ data: [], backgroundColor: [], borderRadius: 2 }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 9 }}, color: '#666' }} }},
                        y: {{ grid: {{ color: '#eee' }}, ticks: {{ font: {{ size: 9 }}, color: '#999', callback: v => Math.round(v*100)+'%' }}, max: 1 }}
                    }}
                }}
            }});
            
            tooltipChart = new Chart(document.getElementById('tooltip-chart'), {{
                type: 'bar',
                data: {{ labels: [], datasets: [{{ data: [], backgroundColor: ['#6366f1', '#8b5cf6', '#a855f7'], borderRadius: 2 }}] }},
                options: {{
                    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{ x: {{ display: false, max: 1 }}, y: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 8 }}, color: '#666' }} }} }}
                }}
            }});
        }}
        
        function update() {{
            document.getElementById('current-layer').textContent = state.layer;
            document.getElementById('layer-slider').value = state.layer;
            document.getElementById('matrix-info').textContent = `Layer ${{state.layer}}`;
            
            entropyChart.data.datasets[0].data = DATA.entropy[state.layer];
            entropyChart.update('none');
            
            document.querySelectorAll('.head-btn').forEach((btn, i) => {{
                btn.classList.toggle('active', i === state.head);
            }});
            
            document.getElementById('primary-label').textContent = `Head ${{state.head}}`;
            renderMatrix('primary-matrix', state.head);
            
            if (state.compare) {{
                document.getElementById('compare-label').textContent = `Head ${{state.compareHead}}`;
                renderMatrix('compare-matrix', state.compareHead);
            }}
            
            updateDist();
        }}
        
        function renderMatrix(id, head) {{
            const container = document.getElementById(id);
            const attn = DATA.attention[state.layer][head];
            const n = DATA.tokens.length;
            
            let html = '<div class="matrix-header">';
            for (let i = 0; i < n; i++) {{
                html += `<span title="${{DATA.tokens[i]}}">${{DATA.tokens[i].trim() || '␣'}}</span>`;
            }}
            html += '</div>';
            
            for (let r = 0; r < n; r++) {{
                html += '<div class="matrix-row">';
                html += `<div class="matrix-row-label" title="${{DATA.tokens[r]}}">${{DATA.tokens[r].trim() || '␣'}}</div>`;
                for (let c = 0; c < n; c++) {{
                    if (c <= r) {{
                        const v = attn[r][c];
                        const hue = 250 - v * 30;
                        const sat = 50 + v * 40;
                        const light = 95 - v * 50;
                        const color = `hsl(${{hue}}, ${{sat}}%, ${{light}}%)`;
                        html += `<div class="matrix-cell" style="background:${{color}}" 
                            onmouseenter="showTip(event,${{r}},${{c}},${{head}})" onmouseleave="hideTip()"></div>`;
                    }} else {{
                        html += `<div class="matrix-cell masked"></div>`;
                    }}
                }}
                html += '</div>';
            }}
            container.innerHTML = html;
        }}
        
        function updateDist() {{
            const row = DATA.attention[state.layer][state.head][state.token];
            distChart.data.datasets[0].data = row;
            distChart.data.datasets[0].backgroundColor = getDistColors(row);
            distChart.update('none');
        }}
        
        function selectHead(h) {{
            if (state.compare && state.head !== h) {{
                state.compareHead = h;
            }} else {{
                state.head = h;
            }}
            update();
        }}
        
        function selectToken(t) {{
            state.token = t;
            document.querySelectorAll('.token-pill').forEach((p, i) => p.classList.toggle('active', i === t));
            updateDist();
        }}
        
        function setCompare(on) {{
            state.compare = on;
            document.getElementById('single-btn').classList.toggle('active', !on);
            document.getElementById('compare-btn').classList.toggle('active', on);
            document.getElementById('compare-wrapper').style.display = on ? 'block' : 'none';
            if (on) update();
        }}
        
        function togglePlay() {{
            state.playing ? stopPlay() : startPlay();
        }}
        
        function startPlay() {{
            state.playing = true;
            document.getElementById('play-btn').textContent = '⏸ Pause';
            const speed = +document.getElementById('speed-select').value;
            state.interval = setInterval(() => {{
                state.layer = (state.layer + 1) % DATA.num_layers;
                update();
            }}, speed);
        }}
        
        function stopPlay() {{
            state.playing = false;
            document.getElementById('play-btn').textContent = '▶ Play';
            if (state.interval) {{ clearInterval(state.interval); state.interval = null; }}
        }}
        
        function showTip(e, r, c, h) {{
            const tip = document.getElementById('tooltip');
            const attn = DATA.attention[state.layer][h];
            const v = attn[r][c];
            
            document.getElementById('tooltip-pair').textContent = `"${{DATA.tokens[r].trim()||'␣'}}" → "${{DATA.tokens[c].trim()||'␣'}}"`;
            document.getElementById('tooltip-value').textContent = (v * 100).toFixed(1) + '%';
            
            const row = attn[r].slice(0, r + 1).map((val, i) => ({{val, i}})).sort((a, b) => b.val - a.val).slice(0, 3);
            tooltipChart.data.labels = row.map(x => DATA.tokens[x.i].trim() || '␣');
            tooltipChart.data.datasets[0].data = row.map(x => x.val);
            tooltipChart.update('none');
            
            document.getElementById('tooltip-list').innerHTML = row.map((x, i) => {{
                const hue = 240 + i * 20;
                return `
                <div class="tooltip-row">
                    <span class="tooltip-rank" style="background:hsl(${{hue}}, 60%, 92%); color:hsl(${{hue}}, 60%, 40%);">${{i+1}}</span>
                    <span class="tooltip-token">${{DATA.tokens[x.i].trim()||'␣'}}</span>
                    <span class="tooltip-bar"><span class="tooltip-bar-fill" style="width:${{x.val*100}}%; background:hsl(${{hue}}, 65%, 55%);"></span></span>
                    <span class="tooltip-pct">${{(x.val*100).toFixed(1)}}%</span>
                </div>
            `}}).join('');
            
            const rect = e.target.getBoundingClientRect();
            let left = rect.right + 8, top = rect.top;
            if (left + 220 > window.innerWidth) left = rect.left - 220;
            if (top + 180 > window.innerHeight) top = window.innerHeight - 190;
            tip.style.left = left + 'px';
            tip.style.top = top + 'px';
            tip.classList.add('visible');
        }}
        
        function hideTip() {{
            document.getElementById('tooltip').classList.remove('visible');
        }}
    </script>
</body>
</html>'''


if __name__ == "__main__":
    visualize_interactive()
