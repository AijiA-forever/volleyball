// 账号体系前端增强：注册 / 密码登录 / 单点登录 / 角色权限
(function () {
  const AUTH_KEY = 'vb_auth';

  function getAuth() {
    try { return JSON.parse(localStorage.getItem(AUTH_KEY) || 'null'); } catch (e) { return null; }
  }
  function setAuth(data) { localStorage.setItem(AUTH_KEY, JSON.stringify(data)); }
  function clearAuth() { localStorage.removeItem(AUTH_KEY); }
  window.vbGetAuth = getAuth;

  const nativeFetch = window.fetch.bind(window);
  async function authFetch(url, opts) {
    opts = opts || {};
    opts.headers = Object.assign({}, opts.headers || {});
    const auth = getAuth();
    if (auth && auth.token) { opts.headers['Authorization'] = 'Bearer ' + auth.token; }
    const res = await nativeFetch(url, opts);
    if (res.status === 401 && getAuth()) { onKicked(); }
    return res;
  }

  window.api = async function (url, opts) {
    const res = await authFetch(url, opts);
    if (res.status === 503) {
      toast('系统维护中，请稍后再试');
      throw new Error('系统维护中');
    }
    if (!res.ok) {
      let msg = '请求失败';
      try { msg = (await res.json()).detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    return res.json();
  };

  async function updateMaintenanceButton() {
    const btn = $('maintenanceBtn');
    if (!btn) return;
    if (!state.user || state.user.role !== 'coach') { btn.style.display = 'none'; return; }
    btn.style.display = '';
    try {
      const st = await window.api('/api/maintenance/status');
      btn.textContent = st.maintenance ? '退出维护模式' : '进入维护模式';
    } catch (e) {}
  }

  window.toggleMaintenance = async function () {
    const btn = $('maintenanceBtn');
    const isOn = btn && btn.textContent.indexOf('退出') >= 0;
    try {
      const r = await window.api('/api/maintenance/' + (isOn ? 'disable' : 'enable'), { method: 'POST' });
      toast(r.message || '操作成功');
      if (btn) btn.textContent = r.maintenance ? '退出维护模式' : '进入维护模式';
    } catch (e) { toast(e.message); }
  };

  function onKicked() {
    clearAuth();
    localStorage.removeItem('vb_user');
    state.user = null;
    state.role = 'student';
    updateBadge();
    applyRoleUI();
    $('loginModal').classList.add('show');
    toast('账号已在其他设备登录，或登录已过期，请重新登录');
  }

  function updateBadge() {
    const badge = $('userBadge');
    if (!badge) return;
    if (state.user) { badge.textContent = (state.user.role === 'coach' ? '教练：' : '学生：') + state.user.username; }
    else { badge.textContent = '未登录，点击进入'; }
  }

  function applyRoleUI() {
    const isCoach = state.role === 'coach';
    document.querySelectorAll('#nav button').forEach(function (b) {
      if (['coachStd', 'report', 'calibration', 'dataset'].indexOf(b.dataset.panel) >= 0) {
        b.style.display = isCoach ? '' : 'none';
      }
    });
    if (!isCoach) {
      const active = document.querySelector('.panel.active');
      if (active && ['panel-coachStd', 'panel-report', 'panel-calibration'].indexOf(active.id) >= 0) {
        showPanel('practice');
      }
    }
    updateMaintenanceButton();
    if (window.refreshDatasetNav) window.refreshDatasetNav();
    const stu = $('practiceStudent');
    if (stu) {
      if (state.user && state.user.role === 'student') { stu.value = state.user.username; stu.disabled = true; }
      else { stu.disabled = false; }
    }
  }

  function buildModal() {
    const box = document.querySelector('#loginModal .modal-box');
    if (!box || box.dataset.authReady === '1') return;
    box.dataset.authReady = '1';
    box.innerHTML = '' +
      '<h3>账号登录</h3>' +
      '<div class="row"><label>账号</label><input type="text" id="loginName" placeholder="学号 / 用户名" style="flex:1"></div>' +
      '<div class="row"><label>密码</label><input type="password" id="loginPassword" style="flex:1"></div>' +
      '<div class="row"><button class="btn" onclick="doLogin()">登录</button><button class="btn gray" onclick="switchAuthTab(1)">注册新账号</button></div>' +
      '<div id="regBox" style="display:none;border-top:1px solid #eee;margin-top:12px;padding-top:12px">' +
      '<h4>注册</h4>' +
      '<div class="row"><label>账号</label><input type="text" id="regName" style="flex:1"></div>' +
      '<div class="row"><label>姓名</label><input type="text" id="regDisplay" style="flex:1"></div>' +
      '<div class="row"><label>密码</label><input type="password" id="regPassword" style="flex:1"></div>' +
      '<div class="row"><label>身份</label><select id="regRole"><option value="student">学生</option><option value="coach">教练</option></select></div>' +
      '<div class="row" id="coachCodeRow" style="display:none"><label>教练注册码</label><input type="text" id="regCoachCode" style="flex:1"></div>' +
      '<div class="row"><button class="btn green" onclick="doRegister()">提交注册</button><button class="btn gray" onclick="switchAuthTab(0)">返回登录</button></div>' +
      '</div>' +
      '<div class="row" style="color:#888;font-size:12px">同一账号只能在一台设备在线；在其他设备登录会使当前设备自动退出。</div>';
    const roleSel = $('regRole');
    if (roleSel) { roleSel.onchange = function () { $('coachCodeRow').style.display = this.value === 'coach' ? '' : 'none'; }; }
  }

  window.switchAuthTab = function (showRegister) {
    $('regBox').style.display = showRegister ? '' : 'none';
  };

  window.doLogin = async function () {
    const username = ($('loginName').value || '').trim();
    const password = $('loginPassword').value || '';
    if (!username || !password) { toast('请输入账号和密码'); return; }
    const fd = new FormData();
    fd.append('username', username);
    fd.append('password', password);
    try {
      const res = await authFetch('/api/login', { method: 'POST', body: fd });
      if (!res.ok) {
        let msg = '登录失败';
        try { msg = (await res.json()).detail || msg; } catch (e) {}
        toast(msg);
        return;
      }
      const data = await res.json();
      setAuth({ token: data.token, user: data.user });
      await afterLogin(data.user);
    } catch (e) { toast(e.message); }
  };

  window.doRegister = async function () {
    const fd = new FormData();
    fd.append('username', ($('regName').value || '').trim());
    fd.append('display_name', ($('regDisplay').value || '').trim());
    fd.append('password', $('regPassword').value || '');
    fd.append('role', $('regRole').value);
    fd.append('coach_code', ($('regCoachCode').value || '').trim());
    try {
      const res = await authFetch('/api/register', { method: 'POST', body: fd });
      if (!res.ok) {
        let msg = '注册失败';
        try { msg = (await res.json()).detail || msg; } catch (e) {}
        toast(msg);
        return;
      }
      toast('注册成功，请返回登录');
      switchAuthTab(0);
    } catch (e) { toast(e.message); }
  };

  window.logout = async function () {
    try { await window.api('/api/logout', { method: 'POST' }); } catch (e) {}
    clearAuth();
    localStorage.removeItem('vb_user');
    state.user = null;
    state.role = 'student';
    updateBadge();
    applyRoleUI();
    $('loginModal').classList.add('show');
    toast('已退出登录');
  };

  window.changePassword = async function () {
    const oldP = prompt('请输入原密码');
    if (!oldP) return;
    const newP = prompt('请输入新密码（至少 4 位）');
    if (!newP) return;
    const fd = new FormData();
    fd.append('old_password', oldP);
    fd.append('new_password', newP);
    try { await window.api('/api/change_password', { method: 'POST', body: fd }); toast('密码已修改'); }
    catch (e) { toast(e.message); }
  };

  async function afterLogin(user) {
    state.user = user;
    state.role = user.role;
    localStorage.setItem('vb_user', JSON.stringify(user));
    updateBadge();
    applyRoleUI();
    $('loginModal').classList.remove('show');
    if (user.role === 'student') { $('practiceStudent').value = user.username; }
    try { await loadPracticeOptions(); } catch (e) {}
    try { await loadStudy(); } catch (e) {}
    try { await loadHistory(); } catch (e) {}
    toast('登录成功：' + (user.role === 'coach' ? '教练' : '学生') + ' ' + user.username);
  }

  function addHeaderButtons() {
    const header = document.querySelector('header .user');
    if (!header || header.dataset.authButtons === '1') return;
    header.dataset.authButtons = '1';
    header.querySelectorAll('span').forEach(function (s) { if (s.textContent === '退出') s.remove(); });
    const maint = document.createElement('span');
    maint.id = 'maintenanceBtn';
    maint.textContent = '进入维护模式';
    maint.style.marginLeft = '12px';
    maint.style.cursor = 'pointer';
    maint.style.display = 'none';
    maint.onclick = window.toggleMaintenance;
    header.appendChild(maint);
    const pwd = document.createElement('span');
    pwd.textContent = '修改密码';
    pwd.style.marginLeft = '12px';
    pwd.style.cursor = 'pointer';
    pwd.onclick = window.changePassword;
    const out = document.createElement('span');
    out.textContent = '退出';
    out.style.marginLeft = '12px';
    out.style.cursor = 'pointer';
    out.onclick = window.logout;
    header.appendChild(pwd);
    header.appendChild(out);
  }

  async function boot() {
    buildModal();
    addHeaderButtons();
    const auth = getAuth();
    if (auth && auth.token) {
      try { const user = await window.api('/api/me'); await afterLogin(user); }
      catch (e) { clearAuth(); updateBadge(); applyRoleUI(); $('loginModal').classList.add('show'); }
    } else {
      updateBadge(); applyRoleUI(); $('loginModal').classList.add('show');
    }
    setInterval(async function () {
      if (!getAuth()) return;
      try { await window.api('/api/me'); } catch (e) {}
    }, 30000);
  }

  if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', boot); }
  else { boot(); }
})();