<script lang="ts">
  import { onMount } from 'svelte';
  export let size = 'md';
  let dots = '.';
  let t: number;
  onMount(() => {
    t = setInterval(() => {
      dots = dots.length >= 3 ? '.' : dots + '.';
    }, 500);
    return () => clearInterval(t);
  });
</script>

<div class="w-full mt-2 mb-2">
  <!-- 상단: 애니메이션 텍스트 -->
  <div class="w-full py-3 px-4 text-sm text-muted-foreground">
    답변을 생각하는 중{dots}
  </div>

  <!-- 하단: 기존 회색 블럭 스켈레톤 -->
  <div class="animate-pulse flex w-full mt-2">
    <div class="{size === 'md' ? 'space-y-2' : 'space-y-1.5'} w-full">
      <div class="{size === 'md' ? 'h-2' : 'h-1.5'} bg-gray-200 dark:bg-gray-600 rounded-sm mr-14"/>
      <div class="grid grid-cols-3 gap-4">
        <div class="{size === 'md' ? 'h-2' : 'h-1.5'} bg-gray-200 dark:bg-gray-600 rounded-sm col-span-2"/>
        <div class="{size === 'md' ? 'h-2' : 'h-1.5'} bg-gray-200 dark:bg-gray-600 rounded-sm col-span-1"/>
      </div>
      <div class="grid grid-cols-4 gap-4">
        <div class="{size === 'md' ? 'h-2' : 'h-1.5'} bg-gray-200 dark:bg-gray-600 rounded-sm col-span-1"/>
        <div class="{size === 'md' ? 'h-2' : 'h-1.5'} bg-gray-200 dark:bg-gray-600 rounded-sm col-span-2"/>
        <div class="{size === 'md' ? 'h-2' : 'h-1.5'} bg-gray-200 dark:bg-gray-600 rounded-sm col-span-1 mr-4"/>
      </div>
      <div class="{size === 'md' ? 'h-2' : 'h-1.5'} bg-gray-200 dark:bg-gray-600 rounded-sm"/>
    </div>
  </div>
</div>
