package plap

#Config: {
  advisor?: #FieldConfig
  sides: {
    advisor: 1
  }
  overlays: {
    model?: {
      [string]: {
        advisor!: #FieldConfig
        [string]: _
      }
    }
  }
}
