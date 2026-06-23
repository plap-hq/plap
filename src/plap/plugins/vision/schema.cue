package plap

#Config: {
  vision?: #FieldConfig
  overlays: {
    model?: {
      [string]: {
        vision!: #FieldConfig
        [string]: _
      }
    }
  }
}
