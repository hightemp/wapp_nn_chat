<script setup>
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="wrapper">
    <div class="app-panel">
      <button v-if="sMode=='chat'" class="btn btn-light">
        <i class="bi bi-chat"></i>
      </button>
    </div>
    <div class="left-panel">
      <template v-if="oMessageDefaultOptions">
        <label>Модель</label>
        <select class="form-control" v-model="oMessageOptions.model">
          <option v-for="sModel in oMessageDefaultOptions.models" :value="sModel">{{ sModel }}</option>
        </select>
        <label>Устройство</label>
        <select class="form-control" v-model="oMessageOptions.device">
          <option v-for="sDevice in oMessageDefaultOptions.devices" :value="sDevice">{{ sDevice }}</option>
        </select>
        <label>Seed {{ oMessageOptions.seed }}</label>
        <input type="range" v-model="oMessageOptions.seed" min="0" max="999999" />
        <label>max_length {{ oMessageOptions.max_length }}</label>
        <input type="range" v-model="oMessageOptions.max_length" min="0" max="10000" />
        <label>top_k {{ oMessageOptions.top_k }}</label>
        <input type="range" v-model="oMessageOptions.top_k" min="0" max="100" />
        <label>temperature {{ oMessageOptions.temperature }}</label>
        <input type="range" v-model="oMessageOptions.temperature" min="0.0" max="1.0" step="0.01" />
        <label>repetition_penalty {{ oMessageOptions.repetition_penalty }}</label>
        <input type="range" v-model="oMessageOptions.repetition_penalty" min="0.0" max="3.0" step="0.01" />
      </template>
    </div>
    <div class="chat-panel">
      <div class="chat-input-panel">
        <textarea class="form-control" v-model="oMessageOptions.message"></textarea>
        <button class="btn btn-light" @click="fnSendMessage">Отправить</button>
      </div>
      <div class="chat-messages-history">
        <div class="chat-messages-history_inner">
          <template v-for="oHistory in aHistoryReverse">
            <div v-if="oHistory.type=='user'" class="message user-message">{{ oHistory.message }}</div>
            <div v-if="oHistory.type=='bot'" class="message bot-message">{{ oHistory.message }}</div>
            <div v-if="oHistory.type=='error'" class="message error-message">{{ oHistory.message }}</div>
          </template>
        </div>
      </div>
    </div>
  </div>
  <div v-show="bShowLoader" class="loader">
    <div class="center">
        <div class="lds-dual-ring"></div>
    </div>
  </div>
</template>

<script>

import API from "./api"

export default {
  computed: {
    aHistoryReverse: {
      get() { return this.aHistory.slice().reverse() }
    }
  },
  data() {
    return {
      sMode: "chat",
      aHistory: [
        // { type: 'user', message: `` },
        // { type: 'bot', message: `` },
      ],
      oMessageOptions: {
        type: "user",
        model: "",
        device: "",
        message: "",
        seed: 0,
        max_length: 600,
        top_k: 1,
        temperature: 0.9,
        repetition_penalty: 2.0
      },
      oMessageDefaultOptions: null,
      bShowLoader: false,
    }
  },

  methods: {
    async fnInit() {
      this.oMessageDefaultOptions = await API.fnGetFilterOptions()
      this.oMessageOptions.model = this.oMessageDefaultOptions.models[0]
      this.oMessageOptions.device = this.oMessageDefaultOptions.devices[0]
    },
    async fnSendMessage() {
      this.bShowLoader = true
      this.aHistory.push({...this.oMessageOptions })
      var oM = await API.fnProcessChatMessage({...this.oMessageOptions })
      this.aHistory.push({ type: "bot", ...oM })
      this.oMessageOptions.message = ""
      this.bShowLoader = false
    }
  },

  created() {
    this.fnInit()
  }
}
</script>