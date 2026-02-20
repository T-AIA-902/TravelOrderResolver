<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { RequestMetric } from '../../api/types'

use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer])

const props = defineProps<{
  metrics: RequestMetric[]
}>()

const option = computed(() => {
  const sorted = [...props.metrics].reverse()
  return {
    tooltip: {
      trigger: 'axis',
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      formatter: (params: any) => {
        const p = params[0]
        const metric = sorted[p.dataIndex as number]
        return `<b>${metric?.model_name ?? '-'}</b><br/>Latence: ${(metric?.duration_s ?? 0) * 1000}ms<br/>${metric?.input_text?.slice(0, 40) ?? ''}...`
      },
    },
    grid: { left: 50, right: 20, top: 20, bottom: 30 },
    xAxis: {
      type: 'category',
      data: sorted.map((_, i) => `#${i + 1}`),
      axisLabel: { fontSize: 10 },
    },
    yAxis: {
      type: 'value',
      name: 'ms',
      axisLabel: { fontSize: 10 },
    },
    series: [
      {
        type: 'line',
        data: sorted.map((m) => +(m.duration_s * 1000).toFixed(0)),
        smooth: true,
        lineStyle: { color: '#3b82f6', width: 2 },
        itemStyle: { color: '#3b82f6' },
        areaStyle: { color: 'rgba(59,130,246,0.08)' },
      },
    ],
  }
})
</script>

<template>
  <div class="rounded-xl border border-gray-200 bg-white p-6">
    <h3 class="mb-4 text-sm font-semibold text-gray-900">Latence par requête</h3>
    <VChart v-if="metrics.length > 0" :option="option" style="height: 250px" autoresize />
    <p v-else class="text-sm text-gray-400">Aucune requête enregistrée</p>
  </div>
</template>
