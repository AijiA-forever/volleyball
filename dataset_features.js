// 教练端：数据集管理与模型重训面板
(function () {
  let built = false;
  let uploadTarget = null;

  function el(tag, html) { const d = document.createElement(tag); if (html !== undefined) d.innerHTML = html; return d; }

  function build() {
    if (built) return;
    built = true;
    const nav = document.getElementById('nav');
    const btn = el('button', '🧪 数据集与训练');
    btn.dataset.panel = 'dataset';
    btn.id = 'datasetNavBtn';
    btn.onclick = showPanel;
    nav.appendChild(btn);

    const main = document.querySelector('main');
    const section = el('section');
    section.id = 'panel-dataset';
    section.className = 'panel';
    section.innerHTML = '' +
      '<div class="card"><h3>数据集管理</h3>' +
      '<div class="row"><input type="text" id="dsName" placeholder="数据集名称">' +
      '<select id="dsTask"><option value="detect">目标检测（排球）</option><option value="pose">姿态估计（关键点）</option></select>' +
      '<button class="btn green" onclick="createDataset()">新建数据集</button>' +
      '<span style="color:#888;font-size:12px">上传图片时可同时上传同名 .txt 标签文件</span></div>' +
      '<div id="dsList"></div></div>' +
      '<div class="card"><h3>训练任务</h3><div id="jobList"></div></div>' +
      '<div class="modal" id="jobLogModal"><div class="modal-box"><h3>训练日志</h3>' +
      '<pre id="jobLogText" style="max-height:60vh;overflow:auto;background:#0d1117;color:#c9d1d9;padding:12px;border-radius:8px;font-size:12px"></pre>' +
      '<div class="row" style="margin-top:10px"><button class="btn gray" onclick="closeModal(\'jobLogModal\')">关闭</button></div>' +
      '</div></div>' +
      '<input type="file" id="dsUploadInput" multiple hidden>';
    main.appendChild(section);

    const input = document.getElementById('dsUploadInput');
    input.onchange = async function () {
      if (!uploadTarget || !input.files.length) return;
      const fd = new FormData();
      for (const f of input.files) fd.append('files', f);
      try {
        const r = await window.api('/api/datasets/' + uploadTarget + '/upload', { method: 'POST', body: fd });
        toast('已上传 ' + r.saved.length + ' 个文件，共 ' + r.items + ' 条数据');
        loadDatasets();
      } catch (e) { toast(e.message); }
      input.value = '';
      uploadTarget = null;
    };
  }

  function showPanel() {
    if (!state.user || state.user.role !== 'coach') { toast('需要教练权限'); return; }
    document.querySelectorAll('#nav button').forEach(b => b.classList.toggle('active', b.dataset.panel === 'dataset'));
    document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'panel-dataset'));
    loadDatasets();
    loadJobs();
  }

  window.refreshDatasetNav = function () {
    const isCoach = state.user && state.user.role === 'coach';
    if (isCoach) { build(); document.getElementById('datasetNavBtn').style.display = ''; }
    else if (built) { document.getElementById('datasetNavBtn').style.display = 'none'; }
  };

  window.createDataset = async function () {
    const fd = new FormData();
    fd.append('name', document.getElementById('dsName').value.trim());
    fd.append('task', document.getElementById('dsTask').value);
    try { await window.api('/api/datasets', { method: 'POST', body: fd }); toast('数据集已创建'); loadDatasets(); }
    catch (e) { toast(e.message); }
  };

  window.deleteDataset = async function (id) {
    if (!confirm('确认删除数据集 ' + id + '？（数据库记录会删除，磁盘文件保留）')) return;
    try { await window.api('/api/datasets/' + id, { method: 'DELETE' }); toast('已删除'); loadDatasets(); }
    catch (e) { toast(e.message); }
  };

  window.uploadDataset = function (id) { uploadTarget = id; document.getElementById('dsUploadInput').click(); };

  window.startTraining = async function (id, task) {
    const epochs = prompt('训练轮数 epochs', '100');
    if (!epochs) return;
    const model = prompt('基础模型', task === 'pose' ? 'yolov8n-pose.pt' : 'yolov8n.pt');
    if (!model) return;
    const imgsz = prompt('输入尺寸 imgsz', '640') || '640';
    const batch = prompt('batch', '8') || '8';
    const fd = new FormData();
    fd.append('task', task); fd.append('model', model); fd.append('epochs', epochs);
    fd.append('imgsz', imgsz); fd.append('batch', batch);
    try {
      const r = await window.api('/api/datasets/' + id + '/train', { method: 'POST', body: fd });
      toast('训练任务已启动 #' + r.job_id + '（后台运行）');
      loadJobs();
    } catch (e) { toast(e.message); }
  };

  window.viewJobLog = async function (id) {
    try {
      const r = await window.api('/api/training_jobs/' + id + '/log');
      document.getElementById('jobLogText').textContent = r.log || '(暂无日志)';
      document.getElementById('jobLogModal').classList.add('show');
    } catch (e) { toast(e.message); }
  };

  window.activateJob = async function (id) {
    if (!confirm('确认用该任务的权重替换当前生产模型？（会自动备份旧权重）')) return;
    try { const r = await window.api('/api/training_jobs/' + id + '/activate', { method: 'POST' }); toast('已启用新模型：' + r.target + '，旧模型备份为 ' + r.backup); }
    catch (e) { toast(e.message); }
  };

  async function loadDatasets() {
    try {
      const list = await window.api('/api/datasets');
      const rows = list.map(d => '<tr><td>' + d.id + '</td><td>' + d.name + '</td><td>' + d.task + '</td><td>' + d.item_count + '</td>' +
        '<td><button class="btn small" onclick="uploadDataset(' + d.id + ')">上传图片</button> ' +
        '<button class="btn small green" onclick="startTraining(' + d.id + ',\'' + d.task + '\')">开始训练</button> ' +
        '<button class="btn small red" onclick="deleteDataset(' + d.id + ')">删除</button></td></tr>').join('');
      document.getElementById('dsList').innerHTML = list.length ?
        '<table><tr><th>ID</th><th>名称</th><th>任务</th><th>数据量</th><th>操作</th></tr>' + rows + '</table>' :
        '<div class="empty">暂无数据集，先创建一个</div>';
    } catch (e) { document.getElementById('dsList').innerHTML = '<div class="empty">' + e.message + '</div>'; }
  }

  async function loadJobs() {
    try {
      const jobs = await window.api('/api/training_jobs');
      const rows = jobs.map(j => '<tr><td>' + j.id + '</td><td>' + j.dataset_id + '</td><td>' + j.task + '</td><td>' + j.status + '</td><td>' + j.updated_at + '</td>' +
        '<td><button class="btn small" onclick="viewJobLog(' + j.id + ')">日志</button> ' +
        (String(j.status).indexOf('success') === 0 ? '<button class="btn small green" onclick="activateJob(' + j.id + ')">启用模型</button>' : '') + '</td></tr>').join('');
      document.getElementById('jobList').innerHTML = jobs.length ?
        '<table><tr><th>ID</th><th>数据集</th><th>任务</th><th>状态</th><th>更新时间</th><th>操作</th></tr>' + rows + '</table>' :
        '<div class="empty">暂无训练任务</div>';
    } catch (e) { document.getElementById('jobList').innerHTML = '<div class="empty">' + e.message + '</div>'; }
  }

  if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', function () { setTimeout(window.refreshDatasetNav, 300); }); }
  else { setTimeout(window.refreshDatasetNav, 300); }
})();