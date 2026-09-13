// 独立前端增强：在标准动作编辑弹窗中支持“当前时间添加关键帧”
(function () {
  const stdVideoInput = document.getElementById('stdVideo');
  if (!stdVideoInput) return;
  const row = stdVideoInput.closest('.row');
  if (!row) return;
  const wrap = document.createElement('div');
  wrap.className = 'row';
  wrap.innerHTML = '<label>关键帧标记</label><video id="stdVideoPreview" controls preload="metadata" style="max-height:160px;display:none"></video><button id="addKeyframeBtn" class="btn small">在当前时间添加关键帧</button><span style="color:#888;font-size:12px">先打开示范视频，拖动到动作阶段再点此按钮</span>';
  row.after(wrap);

  function syncMedia() {
    const preview = document.getElementById('stdVideoPreview');
    const oldBox = document.getElementById('stdMediaPreview');
    if (!preview || !oldBox) return;
    const oldVideo = oldBox.querySelector('video');
    if (oldVideo && oldVideo.src) { preview.src = oldVideo.src; preview.style.display = ''; }
    else { preview.removeAttribute('src'); preview.style.display = 'none'; }
  }

  stdVideoInput.addEventListener('change', function () {
    const preview = document.getElementById('stdVideoPreview');
    if (this.files && this.files[0]) { preview.src = URL.createObjectURL(this.files[0]); preview.style.display = ''; }
  });

  const originalOpen = window.openStdEdit;
  window.openStdEdit = async function (id) {
    if (originalOpen) await originalOpen(id);
    setTimeout(syncMedia, 0);
  };

  window.addKeyframe = function () {
    const v = document.getElementById('stdVideoPreview');
    const ta = document.getElementById('stdParams');
    if (!v || !ta || !v.src || !v.currentTime) { toast('请先打开并播放示范视频'); return; }
    let params = {};
    try { params = JSON.parse(ta.value || '{}') || {}; } catch (e) { params = {}; }
    params.keyframes = params.keyframes || [];
    const label = prompt('关键帧名称（例如：准备姿势 / 击球瞬间）', '关键帧');
    params.keyframes.push({ time: Math.round(v.currentTime * 100) / 100, label: label || '关键帧' });
    params.keyframes.sort(function (a, b) { return a.time - b.time; });
    ta.value = JSON.stringify(params, null, 2);
    toast('已添加关键帧');
  };
})();
// 练习页：切换动作类型时自动选中第一个同类型标准
(function () {
  const actionSel = document.getElementById('practiceAction');
  const stdSel = document.getElementById('practiceStandard');
  if (!actionSel || !stdSel) return;
  actionSel.addEventListener('change', function () {
    const list = (typeof state !== 'undefined' && state.standards) || [];
    const first = list.find(function (s) { return s.action_type === actionSel.value; });
    stdSel.value = first ? String(first.id) : '';
  });
})();