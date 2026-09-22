'use strict';
const dialog = document.getElementById('image-dialog');
const dialogImage = document.getElementById('dialog-image');
const dialogCaption = document.getElementById('dialog-caption');
let previousFocus;
document.querySelectorAll('[data-image]').forEach(button => {
  button.addEventListener('click', () => {
    previousFocus = button;
    dialogImage.src = button.dataset.image;
    dialogImage.alt = button.dataset.caption;
    dialogCaption.textContent = button.dataset.caption;
    dialog.showModal();
  });
});
document.getElementById('close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
dialog.addEventListener('close', () => previousFocus?.focus());
const video = document.getElementById('demo-video');
const videoStatus = document.getElementById('video-status');
document.querySelectorAll('[data-seek]').forEach(button => {
  button.addEventListener('click', async () => {
    video.currentTime = Number(button.dataset.seek);
    try { await video.play(); videoStatus.textContent = '正在播放所选章节'; }
    catch { videoStatus.textContent = '请点击视频中的播放按钮开始观看'; }
    video.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' });
  });
});
